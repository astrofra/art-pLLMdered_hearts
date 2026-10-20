# Voix du LLM et reprise du magnéto

Le point de vue retenu est celui du **LLM à la première personne**, en anglais. Son expression doit être concrète et conversationnelle, proche de celle du témoignage. Il attribue les souvenirs à Amy : il n'incarne ni l'autrice ni l'héroïne.

Le prompt demande un rapprochement explicite entre un détail du jeu et un souvenir documenté, développé sur deux phrases. Le contexte compact contient désormais 191 mots : huit fiches avec des exemples plus précis, notamment le coffre sous le lit et les attentes suscitées par les objets familiers. Le jeu abstrait de candidature est explicitement distingué de Plundered Hearts. Les deux derniers commentaires restent disponibles pour limiter les répétitions.

Pendant une vidéo, le déclenchement reste fixé à plus de 150 mots, après affichage complet d'une sortie de jeu. Quand la durée estimée de l'extrait et la marge sont écoulées, le prochain retour au prompt demande un **commentaire neuf, même sous 150 mots**. Seul ce commentaire tout juste affiché peut être vectorisé et choisir le prochain extrait, avant l'envoi d'une nouvelle commande. Les commentaires déjà affichés pendant la vidéo ne sont plus réutilisés plus tard. Une réponse vide conserve le buffer ; après une erreur de vectorisation ou d'écriture, le prochain prompt libre demande de nouveau un commentaire frais.

## Vérifications

- Les **32 tests** du projet passent : seuil pendant la vidéo, nouveau commentaire sous le seuil à l'arrêt, ordre affichage / vectorisation / demande / commande, réponses vides, erreurs d'embedding et d'écriture, mémoire et redémarrages.
- `results.json` conserve huit réponses avec le point de vue LLM, les prompts exacts, les historiques et la mémoire glissante de deux commentaires. Les huit réponses sont complètes et utilisent la première personne. Ce rejeu exploratoire n'est pas une évaluation indépendante de la qualité des rapprochements ; certaines analogies restent larges et des réponses dépassent les deux phrases demandées.
- `runtime-final/` conserve un essai réel Frotz/Ollama de trois commandes avec la dernière consigne, sans renderer ni viewer : un commentaire et une demande vidéo. Les événements vérifient qu'aucune nouvelle sortie ou commande du jeu ne s'intercale entre le commentaire affiché et la demande vidéo.
- Les essais antérieurs (`results-initial.json`, `results-author-viewpoint.json`, `results-llm-draft.json` et `runtime/`) conservent des variantes abandonnées. En particulier, les essais où le modèle s'attribuait les souvenirs de l'autrice ne correspondent pas au comportement retenu.

`check_voice.py` relance le rejeu avec le prompt courant sur huit historiques de `source-session.jsonl`, copie de la session utilisateur `llm_out/commentary_20261020_085931_9632.jsonl`. Cette copie permet de reproduire les essais sans dépendre des journaux locaux. Il écrase `results.json` ; préserver ce fichier sous un autre nom pour comparer une nouvelle révision. Il utilise les dépendances existantes, ne lance pas le jeu et n'écrit aucune demande vidéo. Les réglages sont Gemma 4 E4B, raisonnement activé, température 0,3, contexte 4096, plafond 2560, graine 42. Les limites restent indicatives : le raisonnement et le texte visible partagent le plafond ; le raisonnement lui-même n'est pas enregistré.

Le lancement habituel reste `3-run-ai-play.bat`. Les classements utilisent toujours le meilleur passage par vidéo parmi les 132 passages, avec la récence et les durées au niveau des 25 clips. La disponibilité du magnéto reste estimée à partir des durées, sans retour de Godot.
