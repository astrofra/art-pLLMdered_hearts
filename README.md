# pLLMdered_hearts

**Installation interactive (ou pas)**

[![Installation hero image](img/hero-image.png)](img/hero-image-hd.png)

## Overview

pLLMdered Hearts est une installation qui fait jouer automatiquement Plundered Hearts (Infocom, 1987) tout en affichant une voix interieure generee par LLM. En parallele, un viewer video diffuse des extraits d'interview d'Amy Briggs selectionnes par similarite d'embeddings avec les commentaires du LLM.

Deux recits se croisent : Lady Dimsford dans le jeu, et Amy Briggs dans son temoignage.

## Architecture

- `src/faketerm.py` pilote `frotz`, envoie une solution pre-ecrite, nettoie la sortie, et rend le texte via un renderer C64.
- Le LLM ne choisit pas les commandes : il commente quand le buffer depasse 500 mots, au retour au prompt, apres l'affichage complet de la sortie du jeu.
- Le buffer conserve toutes les commandes et sorties depuis le dernier commentaire reussi. Le commentaire s'affiche immediatement, meme pendant un extrait d'Amy.
- Python garde le dernier commentaire disponible. Une fois l'extrait courant termine selon `duration_sec` et une marge de 3 secondes, ce commentaire est embarque (`ollama.embeddings`) puis compare a `assets/abriggs-itw-embeddings.json` pour choisir le prochain clip video.
- Le choix est ecrit dans `llm_out/` via un fichier timestamp. La synchronisation repose uniquement sur les durees ; Godot ne renvoie aucun etat a Python.
- La boucle redemarre apres la derniere commande pour un fonctionnement continu.
- `godot-viewer/` lit `llm_out/`, met en file les videos, et joue du bruit (noise) quand la file est vide.

## Donnees et scripts

- `src/embed_vtt.py` genere `assets/abriggs-itw-embeddings.json` a partir des sous-titres `.txt` (hors `-fr`), et ajoute `sequence_title`.
- `src/translate_subtitles.py` produit les sous-titres `-fr.txt` avec contexte.
- `src/compute_itw_durations.py` calcule `duration_sec` depuis les timecodes de sous-titres.

## Execution

- Prerequis : Python 3.10 ou plus recent, `frotz.exe` a la racine (ou `frotz` dans le PATH), ROM `roms/PLUNDERE.z3`, et Ollama en fonctionnement.
- Les deux modeles requis sont `gemma4:e4b` et `embeddinggemma:300m`. Les installer si besoin avec `ollama pull gemma4:e4b` et `ollama pull embeddinggemma:300m`.
- Premiere installation : lancer `1-install-python-reqs.bat`. Il cree un environnement Python local `.venv` et y installe les dependances, sans modifier les autres projets.
- Lancement habituel : double-cliquer sur `3-run-ai-play.bat`. Le batch fonctionne aussi depuis un autre repertoire et utilise `.venv` en priorite.
- Les commentaires du LLM, les libelles et les titres des extraits sont en anglais par defaut, pour la demonstration a Loughborough. La variable `COMMENTARY_LANGUAGE` permet de choisir une autre langue pour les commentaires (par exemple `French`).
- Le rendu C64 et le viewer `bin/itw-viewer.exe` sont lances ensemble. Echap ou Ctrl+C arrete le pilote et ses processus enfants. Echap dans le viewer seul ferme sa fenetre.
- Le viewer peut tourner seul et attend les nouvelles demandes dans `llm_out/`. Les demandes des sessions precedentes sont conservees et ignorees au demarrage.
- La variable d'environnement `LLM_OUT_OVERRIDE` permet de choisir un autre dossier de demandes, aussi bien pour Python que pour le viewer exporte. Le pilote transmet automatiquement le chemin absolu au viewer.

Le code reste compatible avec la syntaxe Python 3.10 ; la remise en etat a ete verifiee avec Python 3.12.5 sous Windows. Les seules dependances Python sont `ollama`, `pexpect` et `pygame` ; `wexpect` n'est pas utilise.

### Diagnostic

Depuis un terminal a la racine du projet :

```bat
3-run-ai-play.bat --check
3-run-ai-play.bat --steps 2 --fast
3-run-ai-play.bat --no-llm --no-viewer --no-renderer --once
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`--check` verifie les fichiers et les modeles sans ouvrir de fenetres. `--steps` limite le nombre de commandes envoyees et `--once` arrete le programme apres le parcours complet. `--fast` supprime les delais de frappe pour les essais. Sans ces options, l'installation tourne en boucle.

Les variables `LLM_MODEL`, `VIDEO_EMBED_MODEL`, `OLLAMA_HOST` et `FROTZ_PATH` permettent de choisir les modeles, le serveur et l'interpreteur. Le modele d'embedding doit correspondre a celui utilise pour produire le catalogue. `LLM_TIMEOUT` (180 secondes) et `EMBED_TIMEOUT` (30 secondes) bornent les requetes ; une erreur de commentaire laisse le jeu continuer.

Les embeddings utilisent le processeur par defaut pour garder la memoire graphique disponible pour Gemma, notamment sur une carte de 8 Go. `VIDEO_EMBED_NUM_GPU=-1` permet a Ollama de choisir automatiquement le placement si la carte dispose de davantage de memoire. Le premier commentaire inclut le temps de chargement des modeles.

Le seuil `COMMENTARY_MIN_WORDS` vaut 500 par defaut : le compteur inclut les commandes et le texte nettoye du jeu. Le programme attend toujours la fin d'une sortie complete avant de verifier ce seuil. Il ne tronque pas le buffer. Une reponse vide conserve l'historique pour la tentative suivante ; les fins de parcours sont conservees et marquees lors du redemarrage.

Les commentaires continuent pendant les videos. Seule la vectorisation attend la fin estimee de l'extrait, avec `VIDEO_END_MARGIN_SEC=3`. Si plusieurs commentaires ont ete produits entre-temps, seul le dernier choisit le prochain extrait. Sans commentaire disponible, le viewer reste sur le bruit video. Les durees du catalogue proviennent des sous-titres et sont deja legerement prudentes ; une duree absente ou invalide utilise 200 secondes.

`LLM_NUM_CTX=4096` et `LLM_NUM_PREDICT=256` configurent le contexte et le plafond de generation. Le raisonnement reste desactive par defaut. `COMMENTARY_LOG_PATH` peut enregistrer un journal JSONL des buffers, prompts, commentaires et nombres de tokens, par exemple :

```bat
set "COMMENTARY_LOG_PATH=documentation\commentary-session.jsonl"
3-run-ai-play.bat
```

Apres modification du viewer, reconstruire `bin/itw-viewer.exe` et son `.pck` depuis Godot avec le preset `Windows Desktop`. Les videos `.ogv`, les sous-titres et les fontes doivent rester inclus dans l'export.

### Comparaison des modeles de commentaire

Les lanceurs `4-test-qwen-4b.bat` et `4-test-gemma-e4b.bat` ouvrent l'installation avec, respectivement, `qwen3.5:4b` et `gemma4:e4b`. Ils acceptent les memes options de diagnostic que le lanceur habituel, par exemple `--steps 2 --fast`. Les modeles doivent etre installes avec `ollama pull` et necessitent un Ollama recent ; les essais du 19 octobre 2026 utilisent la version 0.40.6.

Le lanceur habituel utilise desormais `gemma4:e4b`, avec `LLM_THINK=0` par defaut pour obtenir directement les commentaires. Les lanceurs de test fixent aussi `LLM_THINK=0`. La variable reste configurable ; pour les autres familles de modeles, son absence conserve leur comportement de raisonnement par defaut. `LLM_DISPLAY_NAME` peut personnaliser le nom affiche ; par defaut, il correspond au modele choisi. Le client Python Ollama est mis a jour via `1-install-python-reqs.bat` pour prendre en charge cette option.

Le comparatif reproductible utilise le prompt de l'installation et douze moments du parcours, figes une fois puis reutilises par tous les modeles :

```bat
.venv\Scripts\python.exe src\ollama_benchmark.py --output-dir documentation\llm-comparison
```

Il compare Ministral 14B aux deux candidats, avec deux graines par scene, une fenetre de 4096 tokens et la meme temperature explicite. Les autres parametres restent ceux du modele distribue par Ollama. EmbeddingGemma 300M utilise le processeur. Le dossier produit contient `scenes.json`, `results.json` et `comparison.html`, avec les commentaires, leurs meilleurs rapprochements video, les durees et le placement memoire. Les indicateurs de mots techniques et de diversite des videos servent a orienter la lecture ; ils ne mesurent pas automatiquement la qualite des associations.

Pour comparer un autre resume sur exactement les memes scenes et consignes, utiliser `--scenes` et `--summary` avec un nouveau dossier de sortie :

```bat
.venv\Scripts\python.exe src\ollama_benchmark.py --scenes documentation\llm-comparison-2026-10-19\scenes.json --summary assets\abriggs-itw-1250-words.txt --output-dir documentation\llm-comparison-2026-10-19-1250
```

Le script remplace seulement le resume dans les prompts sauvegardes et enregistre son empreinte. Les prompts archives conservent leurs consignes d'origine, meme apres une evolution de l'installation. Ce choix concerne le benchmark ; le lancement habituel conserve son resume de 750 mots.

Le comparatif du raisonnement de Gemma se relance avec :

```bat
.venv\Scripts\python.exe documentation\llm-thinking-comparison-2026-10-19\run_comparison.py
```

Il reutilise les douze scenes du premier benchmark et compare les modes sans raisonnement (256 et 1024 tokens) et avec raisonnement (1024 tokens), sur deux graines. Les modes sont echauffes, l'ordre des conditions est equilibre et les temps de lecture du prompt et de generation sont enregistres separement. Si le raisonnement atteint le plafond de 1024 tokens, une serie complete a 2048 tokens est ajoutee. Le rapport `comparison.html` et les donnees restent dans ce dossier ; les reglages de l'installation ne sont pas modifies par cet essai.

Le profil du nouveau buffer est dans `documentation/commentary-buffer-analysis/`. `profile_buffers.py` reconstitue les historiques depuis le parcours Frotz capture ; `--measure-tokens` compte leurs tokens avec Gemma. Les comparatifs precedents restent des archives du prompt et du rythme anterieurs.

## Notes

- `llm_out/` est l'interface entre les processus (ne pas y ecrire a la main).
- Le walkthrough est fixe; le LLM ne change pas le gameplay.
- Les videos sont choisies par similarite semantique, pas par ordre chronologique.

## Credits

- Amy Briggs interview [by Jason Scott and the "Get Lamp" team](https://archive.org/details/getlamp-interviews?tab=about), licensed Creative Commons Attribution ShareAlike.
