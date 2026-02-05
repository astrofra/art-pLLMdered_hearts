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

- Prerequis : `frotz`, ROM `roms/PLUNDERE.z3`, `ollama` (modeles `ministral-3:14b` et un modele d'embedding).
- Lancer : `python src/faketerm.py` (le viewer Godot peut etre lance par l'exe dans `bin/itw-viewer.exe`).
- Le viewer peut tourner seul, mais il attend des fichiers dans `llm_out/`.
- Pour l'executable Godot, utiliser `LLM_OUT_OVERRIDE` dans `godot-viewer/main.gd` si le chemin de `llm_out/` n'est pas relatif a l'exe.

## Notes

- `llm_out/` est l'interface entre les processus (ne pas y ecrire a la main).
- Le walkthrough est fixe; le LLM ne change pas le gameplay.
- Les videos sont choisies par similarite semantique, pas par ordre chronologique.

## Credits

- Amy Briggs interview [by Jason Scott and the "Get Lamp" team](https://archive.org/details/getlamp-interviews?tab=about), licensed Creative Commons Attribution ShareAlike.
