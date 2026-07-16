#!/usr/bin/env python3
"""
Prepare, commit and push the current Git changes with an Ollama-generated
commit message.

Default flow:
1. optionally build limited repository context for the LLM;
2. stage every change with `git add -A`, including deleted files;
3. unstage excluded files (for example `.~lock.*`) and oversized files;
4. generate one concise comment per changed file from its isolated diff;
5. generate a global commit message from those comments;
6. commit and run `git push`.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


DEFAULT_MODEL = "ministral-3:14b"
DEFAULT_MAX_DIFF_CHARS = 45000
DEFAULT_MAX_FILE_MB = 90.0
DEFAULT_LOG_COUNT = 12
MAX_COMMIT_MESSAGE_ATTEMPTS = 2
MIN_COMMIT_SUMMARY_ALNUM_CHARS = 8
MIN_COMMIT_MESSAGE_ALNUM_CHARS = 24
LLM_EXCLUDED_ROOTS = (
    PurePosixPath("obsidian/these/.obsidian"),
)


@dataclass(frozen=True)
class ChangedFile:
    status: str
    path: str
    old_path: str | None = None


@dataclass(frozen=True)
class OversizedFile:
    path: str
    size_bytes: int
    old_path: str | None = None


@dataclass(frozen=True)
class PromptContext:
    git_log: str = ""
    outline: str = ""


class CommandError(RuntimeError):
    pass


def decode_output(output: bytes) -> str:
    return output.decode("utf-8", errors="replace").strip()


def run_git(args: list[str], cwd: Path, *, raw: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        command = "git " + " ".join(args)
        stderr = decode_output(result.stderr)
        raise CommandError(f"Echec de `{command}`:\n{stderr}")
    if raw:
        return result.stdout
    return decode_output(result.stdout)


def run_git_for_display(args: list[str], cwd: Path) -> str:
    # Git reports some success information (notably `git push`) on stderr.
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = decode_output(result.stdout)
    if result.returncode != 0:
        command = "git " + " ".join(args)
        if output:
            raise CommandError(f"Echec de `{command}`:\n{output}")
        raise CommandError(f"Echec de `{command}`.")
    return output


def git_root(start: Path) -> Path:
    return Path(run_git(["rev-parse", "--show-toplevel"], start)).resolve()


def require_ollama():
    try:
        import ollama  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Le paquet Python `ollama` est introuvable pour l'interpreteur "
            f"`{sys.executable}`. Installe-le avec: "
            f"{sys.executable} -m pip install ollama"
        ) from exc
    return ollama


def decode_path(value: bytes) -> str:
    return value.decode("utf-8", errors="surrogateescape")


def parse_name_status(raw_status: bytes) -> list[ChangedFile]:
    parts = [part for part in raw_status.split(b"\0") if part]
    entries: list[ChangedFile] = []
    index = 0
    while index < len(parts):
        status = decode_path(parts[index])
        index += 1
        if status.startswith(("R", "C")):
            if index + 1 >= len(parts):
                raise RuntimeError("Sortie `git diff --name-status -z` inattendue.")
            old_path = decode_path(parts[index])
            new_path = decode_path(parts[index + 1])
            index += 2
            entries.append(ChangedFile(status=status, path=new_path, old_path=old_path))
        else:
            if index >= len(parts):
                raise RuntimeError("Sortie `git diff --name-status -z` inattendue.")
            path = decode_path(parts[index])
            index += 1
            entries.append(ChangedFile(status=status, path=path))
    return entries


def status_label(status: str) -> str:
    primary = status[0] if status else "?"
    labels = {
        "A": "ajout",
        "M": "modification",
        "D": "suppression",
        "R": "renommage",
        "C": "copie",
        "T": "changement de type",
        "U": "conflit",
    }
    if status.startswith(("R", "C")) and len(status) > 1:
        return f"{labels.get(primary, status)} ({status[1:]}%)"
    return labels.get(primary, status)


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    head = max_chars // 2
    tail = max_chars - head
    omitted = len(text) - max_chars
    marker = f"\n\n[... diff tronque: {omitted} caracteres omis ...]\n\n"
    return text[:head] + marker + text[-tail:], True


def recent_git_log(repo: Path, count: int) -> str:
    try:
        return run_git(
            [
                "log",
                f"-{count}",
                "--date=short",
                "--pretty=format:%h %ad %s",
            ],
            repo,
        )
    except CommandError:
        return "(historique git indisponible)"


def changed_directories(changed_files: list[ChangedFile]) -> list[str]:
    return sorted(
        {
            str(PurePosixPath(item.path).parent)
            for item in changed_files
            if str(PurePosixPath(item.path).parent) != "."
        }
    )


def repository_outline(repo: Path, changed_files: list[ChangedFile]) -> str:
    tracked = run_git(["ls-files", "-z"], repo, raw=True)
    paths = [decode_path(part) for part in tracked.split(b"\0") if part]
    first_level: dict[str, int] = {}
    second_level: dict[str, int] = {}
    for path in paths:
        segments = path.split("/")
        if not segments:
            continue
        first_level[segments[0]] = first_level.get(segments[0], 0) + 1
        if len(segments) > 1:
            key = "/".join(segments[:2])
            second_level[key] = second_level.get(key, 0) + 1

    changed_dirs = changed_directories(changed_files)

    top = ", ".join(f"{name} ({count})" for name, count in sorted(first_level.items()))
    touched = "\n".join(f"- {path}" for path in changed_dirs[:30])
    if len(changed_dirs) > 30:
        touched += f"\n- ... {len(changed_dirs) - 30} autres dossiers"

    frequent_second_level = sorted(
        second_level.items(), key=lambda item: item[1], reverse=True
    )[:20]
    second = "\n".join(f"- {name}/ ({count})" for name, count in frequent_second_level)

    return textwrap.dedent(
        f"""
        Racine du depot:
        {repo}

        Entrees de premier niveau suivies:
        {top or "(aucune)"}

        Principaux sous-dossiers suivis:
        {second or "(aucun)"}

        Dossiers touches par ce commit:
        {touched or "(racine seulement)"}
        """
    ).strip()


def should_include_git_log(changed_files: list[ChangedFile]) -> bool:
    return len(changed_files) >= 4 and len(changed_directories(changed_files)) >= 2


def should_include_outline(changed_files: list[ChangedFile]) -> bool:
    return len(changed_files) >= 2 or any(
        str(PurePosixPath(item.path).parent) != "." for item in changed_files
    )


def build_prompt_context(
    repo: Path, changed_files: list[ChangedFile], log_count: int
) -> PromptContext:
    outline = repository_outline(repo, changed_files) if should_include_outline(
        changed_files
    ) else ""
    git_log = (
        recent_git_log(repo, log_count)
        if log_count > 0 and should_include_git_log(changed_files)
        else ""
    )
    return PromptContext(git_log=git_log, outline=outline)


def staged_changed_files(repo: Path) -> list[ChangedFile]:
    raw_status = run_git(["diff", "--cached", "--name-status", "-z"], repo, raw=True)
    return parse_name_status(raw_status)


def candidate_paths_for_changed_file(changed_file: ChangedFile) -> list[str]:
    paths = [changed_file.path]
    if changed_file.old_path:
        paths.append(changed_file.old_path)
    return paths


def is_commit_excluded_path(path: str) -> bool:
    return Path(path).name.startswith(".~lock.")


def is_commit_excluded_file(changed_file: ChangedFile) -> bool:
    return any(
        is_commit_excluded_path(path)
        for path in candidate_paths_for_changed_file(changed_file)
    )


def is_llm_excluded_path(path: str) -> bool:
    git_path = PurePosixPath(path)
    return any(
        git_path == excluded_root or excluded_root in git_path.parents
        for excluded_root in LLM_EXCLUDED_ROOTS
    )


def is_llm_excluded_file(changed_file: ChangedFile) -> bool:
    return any(
        is_llm_excluded_path(path)
        for path in candidate_paths_for_changed_file(changed_file)
    )


def split_llm_eligible_files(
    changed_files: list[ChangedFile],
) -> tuple[list[ChangedFile], list[ChangedFile]]:
    llm_eligible: list[ChangedFile] = []
    llm_excluded: list[ChangedFile] = []
    for changed_file in changed_files:
        if is_llm_excluded_file(changed_file):
            llm_excluded.append(changed_file)
        else:
            llm_eligible.append(changed_file)
    return llm_eligible, llm_excluded


def bytes_from_mb(value: float) -> int:
    return int(value * 1024 * 1024)


def human_size(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.1f} Mo"


def find_oversized_staged_files(
    repo: Path, changed_files: list[ChangedFile], max_bytes: int
) -> list[OversizedFile]:
    oversized: list[OversizedFile] = []
    for changed_file in changed_files:
        if changed_file.status.startswith("D"):
            continue

        absolute_path = repo / changed_file.path
        try:
            if not absolute_path.is_file():
                continue
            size_bytes = absolute_path.stat().st_size
        except OSError:
            continue

        if size_bytes > max_bytes:
            oversized.append(
                OversizedFile(
                    path=changed_file.path,
                    size_bytes=size_bytes,
                    old_path=changed_file.old_path,
                )
            )
    return oversized


def unstage_paths(repo: Path, paths: list[str]) -> None:
    if not paths:
        return

    with tempfile.NamedTemporaryFile("wb", delete=False) as pathspec_file:
        pathspec_path = Path(pathspec_file.name)
        for path in paths:
            pathspec_file.write(path.encode("utf-8", errors="surrogateescape"))
            pathspec_file.write(b"\0")

    try:
        run_git(
            [
                "restore",
                "--staged",
                "--pathspec-from-file",
                str(pathspec_path),
                "--pathspec-file-nul",
            ],
            repo,
        )
    finally:
        try:
            pathspec_path.unlink()
        except FileNotFoundError:
            pass


def exclude_commit_excluded_staged_files(
    repo: Path, changed_files: list[ChangedFile]
) -> list[ChangedFile]:
    excluded_files = [
        changed_file
        for changed_file in changed_files
        if is_commit_excluded_file(changed_file)
    ]
    paths_to_unstage: list[str] = []
    for item in excluded_files:
        paths_to_unstage.extend(candidate_paths_for_changed_file(item))

    unstage_paths(repo, paths_to_unstage)
    return excluded_files


def exclude_oversized_staged_files(
    repo: Path, changed_files: list[ChangedFile], max_file_mb: float
) -> list[OversizedFile]:
    max_bytes = bytes_from_mb(max_file_mb)
    oversized = find_oversized_staged_files(repo, changed_files, max_bytes)
    paths_to_unstage: list[str] = []
    for item in oversized:
        if item.old_path:
            paths_to_unstage.append(item.old_path)
        paths_to_unstage.append(item.path)

    unstage_paths(repo, paths_to_unstage)
    return oversized


def format_oversized_files(oversized_files: list[OversizedFile]) -> str:
    lines = [
        f"- {item.path} ({human_size(item.size_bytes)})" for item in oversized_files[:20]
    ]
    if len(oversized_files) > 20:
        lines.append(f"- ... {len(oversized_files) - 20} autre(s) fichier(s)")
    return "\n".join(lines)


def staged_diff_for_file(repo: Path, changed_file: ChangedFile) -> str:
    pathspec = changed_file.path
    diff = run_git(["diff", "--cached", "--", pathspec], repo)
    if diff:
        return diff
    if changed_file.old_path:
        return run_git(["diff", "--cached", "--", changed_file.old_path], repo)
    return ""


def extract_ollama_content(response) -> str:
    message = getattr(response, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()

    if isinstance(response, dict):
        nested = response.get("message")
        if isinstance(nested, dict):
            content = nested.get("content")
            if isinstance(content, str):
                return content.strip()
        content = response.get("response")
        if isinstance(content, str):
            return content.strip()

    raise RuntimeError("Reponse Ollama inattendue: impossible d'extraire le texte.")


def ollama_chat(ollama, model: str, messages: list[dict[str, str]]) -> str:
    try:
        response = ollama.chat(
            model=model,
            messages=messages,
            options={
                "temperature": 0.2,
                "top_p": 0.9,
            },
        )
    except Exception as exc:  # Ollama exposes different exception classes by version.
        raise RuntimeError(
            f"Appel Ollama impossible avec le modele `{model}`. "
            "Verifie que Ollama tourne et que le modele est installe "
            f"(`ollama pull {model}`)."
        ) from exc
    return extract_ollama_content(response)


def format_prompt_section(title: str, content: str) -> str:
    content = content.strip()
    if not content:
        return ""
    return f"{title}:\n{content}"


def per_file_prompt(
    *,
    git_log: str,
    outline: str,
    changed_file: ChangedFile,
    diff: str,
    truncated: bool,
) -> list[dict[str, str]]:
    old_path = f"\nAncien chemin: {changed_file.old_path}" if changed_file.old_path else ""
    truncation_note = (
        "\nLe diff a ete tronque au milieu; signale uniquement les changements visibles."
        if truncated
        else ""
    )
    sections = [
        format_prompt_section("Historique recent du depot", git_log),
        format_prompt_section("Structure utile du depot", outline),
        textwrap.dedent(
            f"""
            Fichier: {changed_file.path}{old_path}
            Statut: {status_label(changed_file.status)} ({changed_file.status})
            {truncation_note}

            Diff isole du fichier:
            ```diff
            {diff or "(diff vide ou fichier binaire sans patch textuel)"}
            ```
            """
        ).strip(),
    ]
    user_content = "\n\n".join(section for section in sections if section)

    return [
        {
            "role": "system",
            "content": (
                "Tu rediges des commentaires de commit en francais. "
                "Sois precis, concis et factuel. Base-toi d'abord sur le chemin du "
                "fichier et le contenu visible du diff. Utilise la structure du depot "
                "uniquement pour lever une ambiguite locale. Utilise l'historique "
                "recent seulement s'il confirme explicitement un element deja visible "
                "dans le diff ou le chemin. N'invente jamais d'objectif de projet, "
                "d'article, d'institution, de corpus, de livraison ou de contexte "
                "absent du diff. "
                "Ne decris pas Git lui-meme. Reponds seulement par 1 a 3 lignes "
                "courtes en texte brut, sans puces ni syntaxe Markdown."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def summary_prompt(
    *,
    git_log: str,
    outline: str,
    file_comments: list[tuple[ChangedFile, str]],
) -> list[dict[str, str]]:
    comments = "\n\n".join(
        f"Fichier: {item.path}\nStatut: {status_label(item.status)} ({item.status})\nCommentaires:\n{comment}"
        for item, comment in file_comments
    )
    sections = [
        format_prompt_section("Historique recent du depot", git_log),
        format_prompt_section("Structure utile du depot", outline),
        format_prompt_section("Commentaires obtenus fichier par fichier", comments),
        textwrap.dedent(
            """
            Produis maintenant le message de commit final.

            Format obligatoire:
            <resume clair et concis>
            <detail concis 1>
            <detail concis 2>

            Contraintes:
            - le message est en texte brut uniquement;
            - aucune syntaxe Markdown: pas de puces, pas de liste numerotee, pas de bloc de code, pas de titre;
            - une idee par ligne apres le resume global;
            - le resume global tient sur une seule ligne;
            - n'ecris jamais la formule "Résumé global du commit :";
            - chaque ligne doit etre complete, sans mot tronque ni prefixe incomplet comme "Aj";
            - les lignes de detail couvrent les changements significatifs sans tout repeter fichier par fichier si des lots sont similaires;
            - conserve les noms propres, dossiers ou corpus importants seulement s'ils sont deja presents dans les commentaires fichier par fichier;
            - n'ajoute pas de bloc de code, pas de titre supplementaire, pas de remarque meta.
            """
        ).strip(),
    ]
    user_content = "\n\n".join(section for section in sections if section)

    return [
        {
            "role": "system",
            "content": (
                "Tu transformes des observations de diff en message de commit francais. "
                "Le resultat doit etre directement utilisable comme message Git en "
                "texte brut. Ne conserve que les informations soutenues par les "
                "commentaires fichier par fichier. Ignore tout contexte historique ou "
                "structurel qui ajouterait des details absents de ces commentaires."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def strip_generated_text_prefix(line: str) -> str:
    stripped = line.strip()
    stripped = re.sub(r"^[-*+]\s+", "", stripped)
    stripped = re.sub(r"^\d+[.)]\s+", "", stripped)
    stripped = stripped.replace("`", "")
    stripped = stripped.replace("*", "")
    return stripped


def alnum_char_count(text: str) -> int:
    return sum(1 for char in text if char.isalnum())


def commit_message_quality_issue(message: str) -> str | None:
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    if not lines:
        return "message vide"

    summary = lines[0]
    if alnum_char_count(summary) < MIN_COMMIT_SUMMARY_ALNUM_CHARS:
        return f"resume trop court: {summary!r}"

    if alnum_char_count("\n".join(lines)) < MIN_COMMIT_MESSAGE_ALNUM_CHARS:
        return "message global trop court"

    return None


def retry_messages_after_short_output(
    messages: list[dict[str, str]], previous_output: str, issue: str
) -> list[dict[str, str]]:
    retry_instruction = textwrap.dedent(
        f"""
        Ta reponse precedente est invalide car elle semble trop courte ou tronquee ({issue}).
        Recommence depuis le debut.

        Contraintes supplementaires:
        - ecris un resume complet, avec plusieurs mots;
        - ne termine jamais sur un fragment de mot;
        - si tu hesites, prefere une formulation generique mais complete.
        """
    ).strip()
    return [
        *messages,
        {"role": "assistant", "content": previous_output},
        {"role": "user", "content": retry_instruction},
    ]


def normalize_commit_message(message: str) -> str:
    summary_label = "Résumé global du commit :"
    normalized_summary_label = "".join(
        char
        for char in unicodedata.normalize("NFKD", summary_label.lower())
        if not unicodedata.combining(char)
    ).rstrip(":")

    def extract_summary_content(line: str) -> str | None:
        line = strip_generated_text_prefix(line)
        normalized = "".join(
            char
            for char in unicodedata.normalize("NFKD", line.lower())
            if not unicodedata.combining(char)
        )
        normalized = normalized.replace("*", "").replace("_", "").replace("`", "")
        if not normalized.strip().startswith(normalized_summary_label):
            return None

        stripped = re.sub(r"^[\s*_`]+", "", line.strip())
        if ":" not in stripped:
            return ""

        content = stripped.split(":", 1)[1].strip()
        content = re.sub(r"^[\s*_`]+", "", content)
        return content

    lines = [line.rstrip() for line in message.strip().splitlines()]
    if not lines:
        raise RuntimeError("Le LLM a renvoye un message de commit vide.")

    leading_summaries: list[str] = []
    index = 0
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue

        summary_content = extract_summary_content(lines[index])
        if summary_content is None:
            break

        leading_summaries.append(summary_content)
        index += 1

    if leading_summaries:
        remaining_lines = [
            strip_generated_text_prefix(line) for line in lines[index:] if line.strip()
        ]
        canonical_summary = next(
            (item.strip() for item in reversed(leading_summaries) if item.strip()),
            "",
        )
        if not canonical_summary and remaining_lines:
            canonical_summary = remaining_lines.pop(0)

        if not canonical_summary:
            raise RuntimeError("Le message de commit final ne contient pas de resume.")

        lines = [canonical_summary]
        if remaining_lines:
            lines.extend(remaining_lines)

    normalized_lines = [strip_generated_text_prefix(line) for line in lines if line.strip()]
    if not normalized_lines:
        raise RuntimeError("Le message de commit final est vide apres normalisation.")

    normalized = "\n".join(normalized_lines) + "\n"
    return normalized


def write_temp_commit_message(repo: Path, message: str) -> Path:
    temp = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        prefix="llm-commit-",
        suffix=".txt",
        dir=repo,
        delete=False,
    )
    with temp:
        temp.write(message)
    return Path(temp.name)


def latest_commit_summary(repo: Path) -> str:
    commit_hash = run_git(["rev-parse", "--short", "HEAD"], repo)
    shortstat = run_git(["show", "--shortstat", "--format=", "-1", "HEAD"], repo)
    lines = [f"Commit cree: {commit_hash}"]
    if shortstat.strip():
        lines.append(shortstat.strip())
    return "\n".join(lines)


def commit_with_message(repo: Path, message: str) -> str:
    message_file = write_temp_commit_message(repo, message)
    try:
        run_git(["commit", "--quiet", "-F", str(message_file)], repo)
        return latest_commit_summary(repo)
    finally:
        try:
            message_file.unlink()
        except FileNotFoundError:
            pass


def push(repo: Path) -> str:
    output = run_git_for_display(["push"], repo)
    if output:
        return output
    return "Push termine sans sortie Git."


def snapshot_index(repo: Path) -> str:
    return run_git(["write-tree"], repo)


def restore_index(repo: Path, tree_hash: str) -> None:
    run_git(["read-tree", tree_hash], repo)


def print_section(title: str, body: str) -> None:
    print(f"\n== {title} ==")
    print(body.strip() if body.strip() else "(vide)")


def format_changed_files(files: list[ChangedFile]) -> str:
    return "\n".join(
        f"- {item.status} {item.old_path + ' -> ' if item.old_path else ''}{item.path}"
        for item in files
    )


def fallback_commit_message_for_excluded_files(
    excluded_files: list[ChangedFile],
) -> str:
    if excluded_files and all(is_llm_excluded_file(item) for item in excluded_files):
        summary = "Mise à jour des fichiers de configuration Obsidian"
        detail = "Fichiers sous .obsidian inclus dans le commit sans analyse LLM."
        return f"{summary}\n{detail}\n"

    return (
        "Mise à jour de fichiers exclus de l'analyse LLM\n"
        "Fichiers exclus de l'analyse LLM inclus dans le commit.\n"
    )


def fallback_commit_message_from_comments(
    file_comments: list[tuple[ChangedFile, str]],
) -> str:
    if not file_comments:
        return "Mise à jour du depot\nMessage de commit genere sans resume LLM fiable.\n"

    if len(file_comments) == 1:
        summary = f"Mise à jour de {file_comments[0][0].path}"
    else:
        summary = f"Mise à jour de {len(file_comments)} fichiers"

    detail_lines: list[str] = []
    for changed_file, comment in file_comments:
        comment_lines = [
            strip_generated_text_prefix(line)
            for line in comment.splitlines()
            if strip_generated_text_prefix(line)
        ]
        if comment_lines:
            detail = comment_lines[0]
        else:
            detail = f"{status_label(changed_file.status).capitalize()} de {changed_file.path}"

        if detail == summary or detail in detail_lines:
            continue
        detail_lines.append(detail)
        if len(detail_lines) == 2:
            break

    return "\n".join([summary, *detail_lines]) + "\n"


def generate_commit_message_from_file_comments(
    ollama,
    model: str,
    *,
    git_log: str,
    outline: str,
    file_comments: list[tuple[ChangedFile, str]],
) -> str:
    messages = summary_prompt(
        git_log=git_log,
        outline=outline,
        file_comments=file_comments,
    )

    for attempt in range(1, MAX_COMMIT_MESSAGE_ATTEMPTS + 1):
        raw_message = ollama_chat(ollama, model, messages)
        try:
            normalized_message = normalize_commit_message(raw_message)
        except RuntimeError as exc:
            issue = str(exc)
        else:
            issue = commit_message_quality_issue(normalized_message)
            if issue is None:
                return normalized_message

        if attempt < MAX_COMMIT_MESSAGE_ATTEMPTS:
            print(f"Message de commit LLM invalide ({issue}). Nouvelle tentative...")
            messages = retry_messages_after_short_output(messages, raw_message, issue)
            continue

        print(
            f"Message de commit LLM invalide ({issue}). "
            "Utilisation d'un fallback deterministe."
        )
        return normalize_commit_message(
            fallback_commit_message_from_comments(file_comments)
        )

    raise RuntimeError("Generation du message de commit impossible.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Genere un message de commit avec Ollama, commit les changements "
            "courants puis lance git push."
        )
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_COMMIT_MODEL", DEFAULT_MODEL),
        help=f"modele Ollama a utiliser (defaut: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--log-count",
        type=int,
        default=DEFAULT_LOG_COUNT,
        help=(
            "nombre maximal de commits recents disponibles pour le LLM quand "
            "le contexte historique est juge utile; 0 desactive "
            f"(defaut: {DEFAULT_LOG_COUNT})"
        ),
    )
    parser.add_argument(
        "--max-diff-chars",
        type=int,
        default=DEFAULT_MAX_DIFF_CHARS,
        help=(
            "taille maximale du diff envoye par fichier; le milieu est tronque "
            f"au-dela (defaut: {DEFAULT_MAX_DIFF_CHARS})"
        ),
    )
    parser.add_argument(
        "--max-file-mb",
        type=float,
        default=DEFAULT_MAX_FILE_MB,
        help=(
            "taille maximale d'un fichier a inclure dans le commit, en Mo; "
            f"les fichiers plus gros sont desindexes (defaut: {DEFAULT_MAX_FILE_MB:g})"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "genere le message et affiche ce qui serait fait, sans commit ni push; "
            "l'index Git initial est restaure en fin d'execution"
        ),
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="cree le commit mais ne lance pas git push",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="teste seulement la detection/staging des changements, sans appeler Ollama",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    start = Path.cwd()
    repo = git_root(start)
    index_snapshot: str | None = None

    print(f"Depot Git: {repo}")

    if args.dry_run:
        index_snapshot = snapshot_index(repo)

    try:
        print("Staging de tous les changements avec `git add -A`...")
        run_git(["add", "-A"], repo)

        changed_files = staged_changed_files(repo)
        commit_excluded_files = exclude_commit_excluded_staged_files(
            repo, changed_files
        )
        if commit_excluded_files:
            print_section(
                "Fichiers exclus du commit",
                format_changed_files(commit_excluded_files),
            )
            changed_files = staged_changed_files(repo)

        oversized_files = exclude_oversized_staged_files(
            repo, changed_files, args.max_file_mb
        )
        if oversized_files:
            print_section(
                f"Fichiers desindexes car > {args.max_file_mb:g} Mo",
                format_oversized_files(oversized_files),
            )
            changed_files = staged_changed_files(repo)

        if not changed_files:
            print("Aucun changement a committer.")
            return 0

        print(f"{len(changed_files)} fichier(s) staged.")
        llm_changed_files, llm_excluded_files = split_llm_eligible_files(changed_files)

        if llm_excluded_files:
            print_section(
                "Fichiers exclus de l'analyse LLM",
                format_changed_files(llm_excluded_files),
            )

        if args.skip_llm:
            print_section(
                "Fichiers staged",
                format_changed_files(changed_files),
            )
            print("Arret demande par --skip-llm.")
            return 0

        commit_message: str
        if not llm_changed_files:
            print("Aucun fichier eligible pour l'analyse LLM.")
            commit_message = normalize_commit_message(
                fallback_commit_message_for_excluded_files(llm_excluded_files)
            )
        else:
            print("Verification de l'API Python Ollama...")
            ollama = require_ollama()
            prompt_context = build_prompt_context(repo, llm_changed_files, args.log_count)
            if prompt_context.git_log:
                print("Contexte LLM: historique Git recent inclus.")
            else:
                print("Contexte LLM: historique Git omis pour limiter les faux positifs.")
            if prompt_context.outline:
                print("Contexte LLM: structure du depot incluse.")
            file_comments: list[tuple[ChangedFile, str]] = []

            for index, changed_file in enumerate(llm_changed_files, start=1):
                print(
                    f"[{index}/{len(llm_changed_files)}] Analyse LLM: {changed_file.path}"
                )
                diff = staged_diff_for_file(repo, changed_file)
                limited_diff, truncated = truncate_text(diff, args.max_diff_chars)
                comment = ollama_chat(
                    ollama,
                    args.model,
                    per_file_prompt(
                        git_log=prompt_context.git_log,
                        outline=prompt_context.outline,
                        changed_file=changed_file,
                        diff=limited_diff,
                        truncated=truncated,
                    ),
                )
                file_comments.append((changed_file, comment))

            print("Synthese globale du commit...")
            commit_message = generate_commit_message_from_file_comments(
                ollama,
                args.model,
                git_log=prompt_context.git_log,
                outline=prompt_context.outline,
                file_comments=file_comments,
            )

        print_section("Message de commit genere", commit_message)

        if args.dry_run:
            print("Dry-run: commit et push non executes.")
            return 0

        print("Creation du commit...")
        commit_output = commit_with_message(repo, commit_message)
        print_section("Git commit", commit_output)

        if args.no_push:
            print("Option --no-push: push non execute.")
            return 0

        print("Push vers le remote configure...")
        push_output = push(repo)
        print_section("Git push", push_output)
        return 0
    finally:
        if index_snapshot is not None:
            restore_index(repo, index_snapshot)
            print("Index Git restaure apres dry-run.")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CommandError, RuntimeError) as exc:
        print(f"\nErreur: {exc}", file=sys.stderr)
        raise SystemExit(1)
