# pLLMdered_hearts

**Installation interactive (ou pas)**

[![Installation hero image](img/hero-image.png)](img/hero-image-hd.png)

## Overview

pLLMdered Hearts est une installation qui fait jouer automatiquement Plundered Hearts (Infocom, 1987) tout en affichant une voix interieure generee par LLM. En parallele, un viewer video diffuse des extraits d'interview d'Amy Briggs selectionnes par similarite d'embeddings avec les commentaires du LLM.

Deux recits se croisent : Lady Dimsford dans le jeu, et Amy Briggs dans son temoignage.

## Architecture

- `src/faketerm.py` pilote `frotz`, envoie une solution pre-ecrite, nettoie la sortie, et rend le texte via un renderer C64.
- Le LLM ne choisit pas les commandes : il commente la situation a chaque prompt.
- Chaque commentaire est embarque (`ollama.embeddings`) puis compare a `assets/abriggs-itw-embeddings.json` pour choisir le prochain clip video (cosine similarity).
- Le choix est ecrit dans `llm_out/` via un fichier timestamp, et un cooldown base sur `duration_sec` evite d'enchainer trop vite.
- La boucle redemarre apres la derniere commande pour un fonctionnement continu.
- `godot-viewer/` lit `llm_out/`, met en file les videos, et joue du bruit (noise) quand la file est vide.

## Donnees et scripts

- `src/embed_vtt.py` genere `assets/abriggs-itw-embeddings.json` a partir des sous-titres `.txt` (hors `-fr`), et ajoute `sequence_title`.
- `src/translate_subtitles.py` produit les sous-titres `-fr.txt` avec contexte.
- `src/compute_itw_durations.py` calcule `duration_sec` depuis les timecodes de sous-titres.

## Execution

- Prerequis : Python 3.10 ou plus recent, `frotz.exe` a la racine (ou `frotz` dans le PATH), ROM `roms/PLUNDERE.z3`, et Ollama en fonctionnement.
- Les deux modeles requis sont `ministral-3:14b` et `embeddinggemma:300m`. Les installer si besoin avec `ollama pull ministral-3:14b` et `ollama pull embeddinggemma:300m`.
- Premiere installation : lancer `1-install-python-reqs.bat`. Il cree un environnement Python local `.venv` et y installe les dependances, sans modifier les autres projets.
- Lancement habituel : double-cliquer sur `3-run-ai-play.bat`. Le batch fonctionne aussi depuis un autre repertoire et utilise `.venv` en priorite.
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

Les embeddings utilisent le processeur par defaut pour garder la memoire graphique disponible pour Ministral, notamment sur une carte de 8 Go. `VIDEO_EMBED_NUM_GPU=-1` permet a Ollama de choisir automatiquement le placement si la carte dispose de davantage de memoire. Le tout premier chargement de Ministral peut prendre une a deux minutes.

Apres modification du viewer, reconstruire `bin/itw-viewer.exe` et son `.pck` depuis Godot avec le preset `Windows Desktop`. Les videos `.ogv`, les sous-titres et les fontes doivent rester inclus dans l'export.

## Notes

- `llm_out/` est l'interface entre les processus (ne pas y ecrire a la main).
- Le walkthrough est fixe; le LLM ne change pas le gameplay.
- Les videos sont choisies par similarite semantique, pas par ordre chronologique.

## Credits

- Amy Briggs interview [by Jason Scott and the "Get Lamp" team](https://archive.org/details/getlamp-interviews?tab=about), licensed Creative Commons Attribution ShareAlike.
