# Rapprocher les commentaires et le témoignage d’Amy Briggs

Ouvrir [comparison.html](comparison.html) : comparateur autonome, en anglais, avec les commentaires, les passages gagnants et leurs timecodes. Les boutons proposent deux cas particulièrement parlants : la phrase vague et le coffre sous le lit.

## Ce que les essais permettent de dire

**Plusieurs vecteurs par vidéo constituent une piste utile, à condition de garder le contexte des passages.** Chaque vidéo reçoit le score de son meilleur passage, sans moyenne. Les 25 vidéos restent les unités de sélection et de récence ; les timecodes servent ici à expliquer le rapprochement, sans changer les coupes vidéo.

- Une phrase volontairement vague, inspirée de l’exemple de l’utilisateur, obtient environ **0,570** avec une phrase isolée parlant de raisons personnelles. Elle n’établit aucun sujet commun précis. Cette phrase de contrôle est inventée ; ce n’est pas une citation d’Amy Briggs.
- Pour le commentaire sur le **coffre glissant sous le lit**, le découpage en groupes de sous-titres visant 70 mots retrouve le passage où Briggs évoque exactement ce coffre et un bug de début de partie. Le vecteur du clip entier sélectionnait un souvenir de bureaux contenant des cartons. Le découpage en phrases complètes perd ce gain particulier : les frontières comptent aussi.
- Sur huit requêtes factuelles écrites pour retrouver des sujets connus, le clip entier et les fenêtres de phrases complètes obtiennent **8/8**, contre **4/8** pour les phrases isolées. Ce petit contrôle ne mesure pas la pertinence artistique d’une association avec le jeu.
- Les fenêtres de phrases complètes visant 70 mots donnent **130 passages pour 25 clips**, médiane **79,5 mots**, étendue **38–190 mots**. Une phrase longue n’est pas coupée. C’est une base à relire, pas une taille optimale démontrée.

La prochaine étape utile serait de relire ces unités : une phrase dont on ne peut identifier « it », « there », « that » ou « the reason » doit souvent être rattachée à sa voisine. Supprimer les hésitations ou imposer un minimum de mots ne restitue pas un sujet manquant. Le corpus source reste intact.

**Changer la voix du commentaire rapproche les vecteurs, sans garantir une meilleure lecture.** Cinq prompts ont produit 120 réponses : 12 historiques identiques, deux graines, 24 réponses par condition. Les variantes plus orales ou explicitement sensibles au témoignage augmentent les cosinus avec l’archive, mais aussi la ressemblance entre commentaires. Certaines transforment de simples réponses d’inventaire en contraintes bureaucratiques imaginées. La variante finale `companion` utilise « wonder » dans **18/24** réponses et revient souvent à la confiance.

Le prompt de production est donc conservé. L’installation utilise maintenant le découpage par groupes de sous-titres visant 70 mots : **132 passages, médiane 75 mots**, dans `assets/abriggs-itw-passages.json`. Ce choix initial retient le cas concret du coffre ; les fenêtres de phrases complètes restent une variante de comparaison. Les deux catalogues utilisent des structures distinctes : le [catalogue expérimental de phrases](catalogue-sentence70.json) a une enveloppe de métadonnées, tandis que le chargeur de production lit une liste de clips. Il n’y a pas de fine-tuning dans ces essais.

## Essai des clusters

Un test supplémentaire applique KMeans aux 132 vecteurs normalisés, dans leurs 768 dimensions : 4, 6, 8 et 10 groupes, graines 42 et 43, 20 initialisations par essai. La requête choisit un ou deux centroïdes par cosinus, puis le meilleur passage parmi les groupes retenus.

Avec un seul groupe autorisé, la même vidéo que la recherche exhaustive n’est conservée que pour **25 à 36 commentaires sur 55**, selon l’essai. Cela mesure une perte du meilleur voisin géométrique, pas une baisse de pertinence artistique démontrée. Plus concrètement, le rapprochement du **coffre est perdu dans 6 configurations sur 8** ; même deux groupes le perdent dans 3/8 configurations. Quelques requêtes factuelles gagnent ou perdent selon le regroupement : ce petit contrôle ne désigne pas une méthode universellement meilleure.

Les silhouettes cosinus observées sont de 0,050 à 0,069 : les partitions ne sont pas très séparées selon cet indicateur. Les passages se recouvrent et ne forment pas 132 observations indépendantes. Avec si peu de candidats, comparer tous les passages est déjà léger ; un filtre préalable ne justifie pas ici les correspondances éliminées. Les clusters restent intéressants pour explorer les thèmes ou visualiser les unités. Aucun filtre de cluster n’est activé dans l’installation. [Méthode et limites de KMeans sur du texte, scikit-learn](https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html).

`clusters.json` conserve les partitions, représentants, diagnostics et choix de chaque requête. `clusters.py` utilise les dépendances d’analyse déjà isolées dans `.tools/projection-python`, sans modifier celles de l’installation et sans nouvel appel aux modèles.

## Protocole et limites

- Corpus : les 25 transcriptions et les 55 commentaires du parcours instrumenté du 19 octobre, identifiés dans [protocol.json](protocol.json). L’essai compare des buffers enregistrés, sans lancer le jeu ni envoyer de requêtes au lecteur vidéo.
- Génération : Gemma 4 E4B, raisonnement activé, température 0,3, contexte 4096, plafond partagé 2560, graines 42 et 43. Toutes les réponses finissent avec `stop`, mais certaines dépassent les deux phrases ou ajoutent une interprétation insuffisamment étayée. Le plus grand total observé est de 1989 tokens de prompt + sortie/raisonnement. Le raisonnement lui-même n’est pas enregistré.
- Les quatre premiers prompts tournent dans un ordre alterné. `companion` est une révision faite après lecture de leurs sorties. Chaque candidat reçoit les deux mêmes commentaires de production précédents, et non sa propre mémoire déroulée. Ce n’est pas une validation indépendante, ni un test de répétition sur une session complète.
- Vectorisation : même EmbeddingGemma 300M, 768 dimensions, CPU, contexte explicite 2048, API moderne `/api/embed`, `truncate=false`. Modèles et empreintes dans [environment.json](environment.json). Les vecteurs modernes peuvent différer légèrement de ceux de l’ancien catalogue, assez pour inverser des scores voisins.
- Huit index : clips entiers ; groupes de sous-titres visant 40, 70, 100 et 140 mots ; une phrase ; trois phrases avec recouvrement ; fenêtres de phrases complètes visant 70 mots avec recouvrement et fusion d’une fin trop courte. La ponctuation des transcriptions spontanées reste imparfaite. Aucun texte de l’archive n’est supprimé.
- Trois formats d’entrée : texte brut, préfixes de similarité symétriques et préfixes de recherche asymétriques. L’étude complète des formats porte sur clips entiers / groupes de 100 mots ; les autres tailles utilisent le texte brut pour isoler le découpage. Le dernier découpage en phrases complètes a été ajouté après observation des premiers résultats.
- Avec les préfixes de similarité, le cosinus moyen entre les familles passe de 0,126 à 0,409, sans amélioration du contrôle factuel déjà à 8/8 sur les clips entiers. Un changement d’échelle ou un rapprochement des nuages projetés ne prouve pas une amélioration du sens. Les formats viennent de la [fiche du modèle original 300M](https://huggingface.co/google/embeddinggemma-300m#prompt-instructions), qui distingue similarité et recherche.
- Le maximum bénéficie d’un plus grand nombre de candidats ; les clips longs ont plus d’occasions de gagner. Aucun score n’est une probabilité de pertinence. Les classements présentés ignorent la récence pour isoler le matching ; une intégration doit continuer de l’appliquer au niveau des 25 clips.
- La [relecture qualitative](review.json) est exploratoire, effectuée par l’agent après lecture des résultats. Il n’y a ni jury indépendant ni annotations de référence pour les 55 scènes. Des travaux de vectorisation CPU ont chevauché la génération GPU : les durées ne sont pas un benchmark contrôlé de latence.

## Fichiers et reproduction

Les scripts de génération utilisent seulement la bibliothèque standard et les dépendances déjà présentes du projet. L’analyse utilise **numpy**, déjà isolé dans `.tools/presentation-python`, pour les produits scalaires ; aucune dépendance n’est ajoutée à l’installation. En l’absence de ce dossier, numpy peut être installé dans un environnement d’analyse séparé.

Depuis la racine du projet, avec Ollama actif pour les étapes non mises en cache :

```powershell
.venv\Scripts\python.exe documentation\commentary-resonance-2026-10-20\experiment.py --stage embeddings
.venv\Scripts\python.exe documentation\commentary-resonance-2026-10-20\experiment.py --stage granularity
.venv\Scripts\python.exe documentation\commentary-resonance-2026-10-20\experiment.py --stage screen
.venv\Scripts\python.exe documentation\commentary-resonance-2026-10-20\experiment.py --stage screen --conditions companion
.venv\Scripts\python.exe documentation\commentary-resonance-2026-10-20\analyse.py --stage retrieval
.venv\Scripts\python.exe documentation\commentary-resonance-2026-10-20\analyse.py --stage granularity
.venv\Scripts\python.exe documentation\commentary-resonance-2026-10-20\analyse.py --stage screen
.venv\Scripts\python.exe documentation\commentary-resonance-2026-10-20\clusters.py
.venv\Scripts\python.exe documentation\commentary-resonance-2026-10-20\report.py
.venv\Scripts\python.exe documentation\commentary-resonance-2026-10-20\checks.py
```

Exécuter les étapes dans cet ordre, sans lancer deux écrivains du cache ou de `screen.json` simultanément. Les réponses et vecteurs existants sont réutilisés. Pour une nouvelle réplication indépendante, copier les scripts et les fichiers de référence dans un nouveau dossier de travail en adaptant les chemins, ou déplacer explicitement les résultats à conserver avant de relancer.

`screen.json` conserve les 120 prompts exacts, les historiques, les deux commentaires précédents, les réponses, les options, les tokens et les durées. `embedding-cache.json.gz` conserve toutes les entrées et tous les vecteurs en pleine précision. `*-inputs.json` conserve les unités et références du cache. `*-evaluation.json`, `generated-passage-matches.json` et `matches.csv` rendent les classements inspectables ; au-delà du top 3, le texte du passage est accessible par son `unit_id` dans les entrées. Le catalogue historique et les sous-titres ne sont pas écrasés. Les scripts d’étude ne changent pas les réglages du lancement ; l’intégration dans `faketerm.py` utilise le nouveau fichier d’assets.

Référence API : [Ollama embed](https://docs.ollama.com/api/embed), notamment l’entrée par lot et le refus de troncature.

Validation de l’intégration : 29 tests du projet, quatre contrôles des données, syntaxe Python 3.10, interactions du comparateur dans Chrome sur ordinateur et mobile. Un essai réel Frotz/Ollama de huit commandes, sans renderer ni viewer, a produit deux commentaires et une demande vidéo journalisée. Le contrôle du coffre a aussi été rejoué avec une nouvelle vectorisation et le catalogue installé : il retrouve l’extrait attendu, avec une faible avance de 0,00267 sur le second. Voir [integration-check.json](integration-check.json) et `runtime-smoke/`.
