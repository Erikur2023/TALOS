# TALOS v2 — navigation autonome sans GPS (Piste 1, verrou B)

Hackathon Robot Humanoïde Défense — Grand Palais, 2-4 décembre 2026.
Cas d'usage : un Unitree G1 traverse forêt, boue et relief **sans GNSS**, se localise en recalant ce qu'il voit sur sa
carte embarquée, replanifie face aux obstacles non cartographiés, rejoint un groupe en tranchée, **dépose le
ravitaillement** et se retire.

> Tout chiffre ci-dessous sort d'un run reproductible : `bash run_all.sh` (≈ 1 h 30 sur 16 cœurs) régénère
> `results/final_*`. Journal des choix de développement : `NOTES_NUIT.md`.

## Architecture

```
capteurs (gyro, odométrie jambes vx/vy, LiDAR, pente IMU) ──► Navigator ──► consigne (vx, wz)
                                                              │  filtre particulaire (x, y, cap, échelles, biais gyro)
                                                              │  A* + replanification + évitement + dégagement
                                                              │  ETA, détection d'anomalies
                     ┌────────────────────────────────────────┴───────────────┐
              sim point (sim.py, rapide)                    G1 sous MuJoCo (mujoco_g1/, politique de marche Unitree)
```

Le **même** `Navigator` (`talos2/navigator.py`) pilote le simulateur ponctuel et le G1 physique sous MuJoCo.

## Résultats

Réussite = robot à < 5 m de l'objectif (vérité terrain) sans chute. IC 95 % de Wilson.

### 1. Simulateur ponctuel, mission complète (navigation + dépôt + retrait)

| | Recalage carte — graines 100-115 | Recalage carte — **graines 400-431 jamais vues** | Odométrie seule (400-431) |
|---|---|---|---|
| Missions réussies | **15/16** (72-99 %) | **27/32** (68-93 %) | 0/32 (0-11 %) |
| Erreur de position, médiane des moyennes | 0,64 m | 0,64 m | 13,6 m |
| Erreur p95, médiane | 2,3 m | 2,1 m | 26,2 m |
| Charge déposée (missions réussies) | 15/15 | 27/27 | — |
| Retrait mené jusqu'au point de repli | 14/15 | 25/27 | — |
| Distance au bord au dépôt (cible 1,2 m), médiane | 1,49 m | 1,49 m | — |
| Distance minimale au bord sur toute la phase finale | 0,77 m | 0,75 m | — |
| Passage à < 0,6 m du bord | 0 | 0 | — |
| Durée / énergie médianes (modèle) | 584 s / 65 Wh | 553 s / 64 Wh | 720 s / 78 Wh |
| Pic de signature thermique, médiane (modèle) | +15,1 K | +15,4 K | +16,3 K |

Échecs sur 400-431 : 3 blocages (400, 417, 430), 1 mission sans chemin (412), 1 arrivée à > 5 m (420, qui n'est pas comptée
ni au dépôt). Les graines 100-115, 200-231 et 300-331 ont servi pendant le développement (voir `NOTES_NUIT.md`) ;
**400-431 est le chiffre de référence.**

### 2. Carte « réelle » (style OSM + MNT) vs carte arbre par arbre — mêmes mondes avec chemins forestiers

| | Carte arbre par arbre | **Carte OSM** (forêt, chemins, MNT, zones humides) | Odométrie seule |
|---|---|---|---|
| Missions réussies | 15/16 | **4/16** (10-50 %) | 0/16 |
| Erreur de position, médiane des moyennes | 0,62 m | **5,1 m** | 9,6 m |
| Erreur p95, médiane | 2,2 m | 22,9 m | 24,0 m |

**Dégradation honnête** : sans arbres individuels, le recalage sur lisières / chemins / relief divise l'erreur moyenne
par ~2 par rapport à l'odométrie seule, mais ne suffit pas pour arriver à 5 m (erreur moyenne 2 à 17 m selon les
missions). Une partie des échecs vient aussi de la navigation : sans arbres connus, chaque tronc est découvert en route
(beaucoup de blocages et de dépassements de temps). Piste : odométrie LiDAR-inertielle (FAST-LIO / Point-LIO) pour la
dérive locale + recalage OSM pour la dérive globale.

### 3. Unitree G1 sous MuJoCo (politique de marche `unitree_rl_gym`, relief atténué à 40 %)

| | LiDAR 2D | **LiDAR 3D + segmentation sol** | Odométrie seule (2D) |
|---|---|---|---|
| Missions réussies | **13/16** (57-93 %) | **10/16** (39-82 %) | 0/16 |
| Chutes | 1 | 2 | 8 |
| Erreur de position, médiane des moyennes | 0,50 m | 1,06 m | 14,4 m |
| Charge déposée (missions réussies) | 13/13 | 10/10 | — |
| Distance au bord au dépôt, médiane (cible G1 : 1,8 m) | 2,08 m | 2,09 m | — |
| Retrait mené jusqu'au point de repli | 12/13 | 9/10 | — |
| Passage à < 0,6 m du bord (distance minimale) | **0/13** (0,89 m) | **0/10** (1,07 m) | — |
| Durée médiane | 973 s | 1 003 s | 861 s |
| Travail mécanique positif mesuré (Σ max(0, τ·q̇)), médiane | 7,1 Wh | 7,3 Wh | 6,7 Wh |
| Segmentation LiDAR 3D : précision / rappel obstacles | — | 0,96 / 0,94 | — |
| Erreur de pente estimée par le nuage, médiane | — | 0,008 m/m | — |

Domaine de consigne de la politique, mesuré (`results/mujoco_calib.json`) : vitesse réelle ≈ 0,9 × consigne jusqu'à
1,1 m/s sur le plat, mais 0,6 m/s commandé ne donne que 0,18 m/s à 11° et 0,08 m/s à 14° → politique entraînée sur sol
plat. D'où le relief atténué (×0,4) et la consigne plafonnée à 0,6 m/s.

### 4. Terrain RÉEL du front ukrainien — imagerie satellite vs carte classique, 12 départs par zone

Deux zones de 500 × 500 m : **forêt de Serebrianka** (pinède brûlée près de Kreminna, tranchées et cratères visibles)
et **Robotyne** (village détruit, prairies, haies, ruines). Relief, végétation, chemins et bâtiments sont réels
(imagerie Esri ~0,5 m, MNT SRTM, OpenStreetMap) ; troncs individuels, cratères non cartographiés, évolution depuis la
prise de vue et **tranchée à ravitailler (position fictive, au centre)** sont simulés. 12 départs sur deux anneaux
(140 m et 210 m). Même vérité terrain pour tous les niveaux : seule la carte embarquée change.
`python -m talos2.real_benchmark` → `results/real/benchmark.json`, `results/real/comparison.png`.

| Carte embarquée | Serebrianka : réussite | erreur méd. | Robotyne : réussite | erreur méd. |
|---|---|---|---|---|
| relevé arbre par arbre (borne haute irréaliste) | 12/12 | 0,17 m | 12/12 | 0,33 m |
| **imagerie satellite** (masque de végétation, datée, décalée ~2,5 m) | **11/12** | 3,7 m | **11/12** | 2,0 m |
| carte OSM (polygones forêt, chemins, bâtiments) | **0/12** | 12,8 m | 8/12 | 4,2 m |
| relief seul (MNT) | 3/12 | 6,7 m | 8/12 | 4,5 m |
| aucune (odométrie seule) | 4/12 | 5,4 m | 7/12 | 3,1 m |

À retenir :
- **L'imagerie satellite est la bonne source** : contours fins des clairières, haies, buissons et ruines, sur lesquels
  le LiDAR se recale.
- **Une carte trop grossière peut être PIRE que pas de carte** : à Serebrianka, le polygone OSM « forêt » couvre aussi
  les clairières. Le filtre les confond avec la forêt et dérive (0/12).
- Le relief seul n'apporte presque rien sur ces terrains plats (14 m de dénivelé sur 500 m, MNT à 30 m).
- Sans bonne carte, les échecs sont surtout des blocages en forêt dense (troncs découverts en route) et, au dépôt,
  des passages trop près du bord (1 à 2 par niveau ; 1 aussi avec la carte satellite à Serebrianka).

Deux réglages pour le terrain réel, trouvés sur le départ n° 1 des deux zones (`NOTES_NUIT.md`) :
- **auto-étalonnage en ligne de la densité LiDAR** (impacts observés / attendus) ;
- **injection de particules bornée à 3 m**.
Sans eux, le filtre divergeait (erreur > 100 m).

**Limite importante** : la vérité terrain est tirée de la **même** image que la carte satellite (+ 12 % de massifs
disparus, décalage, troncs aléatoires). Le résultat « satellite » est donc optimiste. Il faudrait un relevé terrain
indépendant (LiDAR aérien) pour le confirmer.

## Carte 3D construite par le robot

L'IHM affiche un nouveau panneau, « Carte 3D construite par le robot ». Il cumule tout ce que le robot a vu, en voxels
de 0,5 m posés à la position **estimée** (une erreur de localisation s'y voit comme un dédoublement) :
- sol coloré par l'altitude ;
- troncs, murs et cratères en orange, jusqu'à la hauteur visible dans le champ du LiDAR 3D (+7°).

Glisser ou molette pour naviguer, double-clic pour le plein écran. Dans le simulateur ponctuel, le nuage 3D est
**approché** à partir du scan 2D et du relief ; le vrai nuage 3D MuJoCo est dans `mujoco_g1/lidar3d.py`. Les runs
« Terrain réel » de l'IHM ont l'image satellite en fond de carte.

## Application TALOS C2 (supervision temps réel, ~100 km de front)

Application de bureau (fenêtre native WebView2 via pywebview ; serveur FastAPI local). **Tout le terrain est réel** :
- **Région** : 96 × 100 km autour de Kramatorsk / Sloviansk / Kostiantynivka / Siversk.
- **Ligne de contact** : 281 km, situation DeepStateMap du 21/11. Seuls les polygones de territoire sont utilisés ;
  les points « unités » de la source sont ignorés.
- **Positions amies : FICTIVES**, générées à 1,5–3 km derrière la ligne (positions avancées) et à 6–11 km
  (points logistiques), étiquetées comme telles partout.

```bash
pip install fastapi uvicorn pywebview rasterio osmium shapely
python -m talos_app.datasets     # ~1,1 Go : WorldCover 10 m, Copernicus DEM 30 m, OSM Ukraine, ligne DeepState (~10 min)
python -m talos_app.app          # fenêtre native   (--browser : dans le navigateur, http://127.0.0.1:8770)
```

**Utilisation** :
1. Choisir A (clic sur la carte ou point logistique) et B (position à ravitailler).
2. Choisir un profil d'itinéraire : rapide / équilibré / sous couvert.
   - Le planificateur stratégique (grille 20 m) refuse les zones occupées et grises et garde 800 m de la ligne.
   - Il pénalise la proximité de la ligne jusqu'à 4 km, ainsi que la pente et l'eau.
   - Il utilise routes, pistes et haies OSM.
3. Choisir le moteur : **rapide** (simulateur 2D, ~7× le temps réel) ou **G1 physique** (MuJoCo, politique Unitree,
   ~1–1,5× le temps réel).
4. Choisir la carte embarquée : satellite / occupation du sol 10 m / relief / aucune.
5. Régler la charge et l'accélération, puis lancer.

**Ce qu'on voit en direct** :
- carte MapLibre (imagerie, topographie, sombre ou hors ligne) avec 20 calques activables : robot estimé et vrai,
  ellipse d'incertitude, particules, impacts LiDAR, chemin local, tuile active ;
- vue MuJoCo en flux vidéo, avec 5 caméras (suivi, tête, côté, dessus, large) ;
- **carte 3D construite** (three.js, voxels dans le repère estimé) ;
- vue tactique 2D de la tuile ;
- télémétrie (12 indicateurs), courbes, journal filtrable et cliquable ;
- rapport JSON téléchargeable.
Chaque panneau peut être agrandi.

**Exécution** :
- L'itinéraire est découpé en tronçons de 240 m. Chaque tronçon se joue dans une tuile de 400 m construite à la
  volée : imagerie Esri, MNT Copernicus, bâtiments et pistes OSM.
- Les cratères sont simulés, avec une densité croissante près de la ligne (2 + 40 × (1 − d/6 km) par ha).
- Sous MuJoCo, l'imagerie satellite est plaquée sur le relief.
- D'une tuile à l'autre, le nuage de particules est transféré. Sous MuJoCo, le robot est reposé dans la nouvelle scène.

**Ajouts de navigation pour le terrain réel** (désactivés par défaut, donc les benchmarks précédents ne changent pas) :
- **cratères = creux** : invisibles pour le LiDAR horizontal, détectés à 8 m par le LiDAR 3D, utilisés pour
  l'évitement mais pas pour le recalage ; on les franchit lentement (×0,4) ;
- **garde de cohérence avec l'odométrie** : un recalage qui s'écarte de plus de 5 m + 8 % de la distance est rejeté ;
- **compas magnétique** (ASSUMPTION : 3° de bruit et une perturbation lente de 2°).
  Sans compas, la dérive du gyroscope (~2°/min) fait diverger l'estimée en terrain agricole ouvert.

**Mission de test** (A → POS-04 FICTIVE, 1,2 km, terres agricoles à 1,5–2 km de la ligne, 5 tuiles, simulateur rapide) :
- carte satellite : **mission accomplie**, arrivée à 2,9 m de B, erreur moyenne 2,7 m, dépôt à 1,45 m du bord,
  retrait mené, 0 contact ;
- odométrie + compas : arrêt à 22 m de B.

Ce sont des **runs isolés, pas un benchmark**.

Sous MuJoCo :
- le premier tronçon passe (erreur 0,5 à 3,7 m) ;
- un essai est tombé dans le tronçon 2, après la repose du robot ;
- la politique de marche reste la limite.

## Entraînement des politiques de marche (RL) — humanoïde G1 et quadrupède Go1

Recette « industrielle » : apprentissage par renforcement massivement parallèle sur GPU.
- **Outils** : PPO de Brax, MuJoCo MJX, environnements MuJoCo Playground 0.0.5, 8 192 robots simulés à la fois sur une
  RTX 5060 Laptop (≈ 165 000 pas de simulation par seconde pour le Go1).
- **Terrain « front ukrainien »** (`rl/terrain.py`) : buttes jusqu'à ~20°, labours, ornières, cratères d'obus, éboulis.
- **Conditions** : charge de 3 kg, randomisation de domaine (frottements, masses, gains), observations bruitées.
- **Curriculum** : sol plat → terrain doux → terrain complet, en reprenant les paramètres de l'étape précédente.

```bash
bash rl/setup_wsl.sh && bash rl/fix_versions.sh     # WSL Ubuntu + GPU NVIDIA : JAX CUDA, MJX 3.3, Brax 0.12, Playground 0.0.5
bash rl/train_all.sh                                # Go1 (~25 min) puis G1 29 ddl (~1 h 30), puis rl/evaluate.py
python rl/eval_cpu.py --trials 8                    # protocole pentes sous MuJoCo CPU (Windows)
```

**Deux échecs instructifs**, conservés dans `results/rl/` :
- **Terrain complet d'emblée** (`go1_ua_direct`, 151 M pas) : la politique apprend à **rester immobile**.
- **Récompenses par défaut de Playground** (`go1_plat_sans_mise_en_forme`, 150 M pas, même sur sol plat) : même optimum
  local « debout », avec une vitesse atteinte d'environ 0 m/s pour une consigne de 1 m/s.

**Correctifs** :
- **Mise en forme de la récompense** (`rl/train.py:SHAPING`) : suivi de vitesse ×2, pénalités de pieds et de posture
  allégées. Résultat : 0,50 m/s pour une consigne de 0,5 et 0,99 m/s pour 1,0, sans chute, en 40 M pas.
- **Curriculum** sol plat → terrain doux → terrain complet.

**Pentes** (`rl/eval_cpu.py`, MuJoCo CPU, plan incliné, consigne 0,5 m/s vers le haut, 15 s, 3 kg, poussée latérale à 5 s,
8 essais par pente) :

| Politique | 0° | 10° | 15° | 20° | 25° |
|---|---|---|---|---|---|
| G1 Unitree pré-entraîné (sol plat) | 0,45 m/s | 0,10 | −0,05 (recule) | **7/8 chutes** | **8/8 chutes** |
| Go1 entraîné sur sol plat (nous) | 0,49 | 0,37 | **8/8 chutes** | 3/8 chutes | 5/8 chutes |
| **Go1 curriculum → terrain UA (nous)** | 0,47 | 0,49 | 0,43 | **0,43, 0 chute** | −0,10, 0 chute |

Les politiques Go1 ont été entraînées en MJX (GPU) puis rejouées en MuJoCo C par notre exécuteur numpy
(`talos2/mujoco_g1/quad.py`) : les actions sont identiques à la 3ᵉ décimale. Le transfert simulateur → simulateur est
vérifié.

**Missions complètes sous MuJoCo** (graines 100–115, relief du monde ×1,0 donc non atténué, recalage LiDAR 2D, dépôt) :

| Robot | Réussites | Chutes |
|---|---|---|
| G1 Unitree pré-entraîné | **0/16** | 14 |
| **Go1 entraîné terrain UA** | **5/16** | 11 |

Sous MuJoCo CPU, il a fallu porter le solveur de contact à 30 itérations : avec le réglage MJX du XML Playground
(1 itération), les pieds s'enfonçaient dans le relief. Les chutes restantes du Go1 viennent de l'écart entre :
- l'entraînement : heightfield de 20 m au pas de 8 cm, en MJX ;
- les missions : heightfield de 300 m au pas de 1 m, en MuJoCo CPU, sur 5 à 10 min de marche.

Piste : entraîner directement sur des tuiles de mission, avec des épisodes plus longs.

**G1 humanoïde 29 ddl (Playground)** : sol plat (100 M pas) → terrain doux (40 M) → terrain complet (60 M).
- **Divergence NaN** avec la randomisation de domaine (~1 robot sur 128) : corrigée par `nan_guard` (`rl/train.py`).
- **Résultat (MJX, 128 robots par pente)** : chutes de 24 % à 0°, 42 % à 10°, 98 % à 20° ; 64/512 réussites sur
  terrain UA inédit, contre 265/512 pour le Go1. Non utilisable en mission.
`rl/train_all.sh` enchaîne ensuite `rl/evaluate.py` : pentes et terrain UA inédit (graines 101-102), avec 5 politiques
comparées en MJX. Les résultats s'affichent automatiquement dans l'onglet ENTRAÎNEMENT et dans les benchmarks
(`results/robots/`).


### Comparatif avec des politiques pré-entraînées publiques (Unitree, NVIDIA)

Rejouées dans **notre** banc MuJoCo CPU (`talos2/mujoco_g1/ext_policies.py`), même protocole, 3 kg :
- **Unitree G1 29 ddl officiel** : `unitree_rl_mjlab`, politique de vitesse v0 (Apache-2.0, © Unitree Robotics) ;
- **NVIDIA GR00T-WholeBodyControl**, contrôleur découplé Balance/Walk (code Apache-2.0, poids sous NVIDIA Open
  Model License).

Les dépôts sont clonés dans `../ext` (variable `TALOS_EXT`).

| Pentes (8 essais) | 0° | 10° | 15° | 20° |
|---|---|---|---|---|
| G1 29 ddl Unitree officiel | 0,45 m/s | 0,19 | 0,00 | **8/8 chutes** |
| G1 29 ddl NVIDIA GR00T | 0,42 | −0,20 (recule) | 3/8 chutes | **8/8 chutes** |
| Go1 curriculum terrain UA (nous) | 0,47 | 0,49 | 0,39 | **0,40, 0 chute** |
| Go1 NVIDIA Isaac Lab terrain accidenté (perceptif) | 0,51 | 0,51 | 0,48 | **0,43, 0 chute** (0,42 à 25°) |

| Missions (graines 100–115, relief ×1,0) | Réussites | Chutes | Bloqués |
|---|---|---|---|
| G1 12 ddl Unitree (unitree_rl_gym) | 0/16 | 14 | 2 |
| G1 29 ddl Unitree officiel | 0/16 | 7 | 9 |
| G1 29 ddl NVIDIA GR00T | 0/16 | 12 | 4 |
| Go1 NVIDIA Isaac Lab terrain accidenté | 0/16 | 16 (après 60–90 s) | 0 |
| **Go1 terrain UA (nous)** | **5/16** | 11 | 0 |

**Lecture** :
- Ces politiques humanoïdes visent le robot réel, une marche naturelle et des sols de type intérieur ; un quadrupède
  est intrinsèquement plus stable.
- Notre propre G1 29 ddl (200 M pas) fait bien pire qu'elles : 24 à 100 % de chutes dès le sol plat.
- **Limite** : leurs actionneurs et contacts sont reproduits d'après leur code (écart simulateur → simulateur
  possible). Une évaluation dans leurs propres environnements reste à faire.
- **Isaac Lab Go1** (poids officiels NVIDIA, `Isaac-Velocity-Rough-Unitree-Go1-v0`, chargés en `weights_only`) :
  - meilleure politique en pente (0,42 m/s à 25°) ;
  - chute dans les 16 missions après 60 à 90 s de marche ;
  - limite : son réseau d'actionneurs (TorchScript) est remplacé par un PD Kp 25 / Kd 0,5.
- Le G1 Isaac Lab (37 ddl, mains Dex3) est écarté : pas de modèle MuJoCo fidèle.
- *Walk These Ways* (Go1, MIT) n'a pas été intégré : ses poids sont au format pickle / TorchScript (exécution de code
  au chargement).

## Application — onglets

OPÉRATION (carte et mission), ROBOT EN DIRECT (grande vue MuJoCo, 3D, tactique, courbes), CARTE 3D (plein écran),
ANALYSE MISSION (6 graphiques), BENCHMARKS (tous les comparatifs avec IC 95 %), ENTRAÎNEMENT (courbes, configuration,
terrains, évaluation), DONNÉES. Moteurs de mission : simulateur rapide, **G1 humanoïde** ou **Go1 quadrupède** sous MuJoCo.

## Limites (à annoncer au jury)

- **Relief atténué à 40 % sous MuJoCo** : la politique Unitree pré-entraînée ne franchit pas les pentes du terrain synthétique.
  Il faut une politique entraînée sur terrain accidenté avant le robot réel.
- **Dépôt sous MuJoCo** : à l'arrêt, le G1 piétine et glisse (~0,1 m/s sur 5° de pente, `calib`), d'où une distance
  de dépôt plus grande pour le G1 (1,8 m au lieu de 1,2 m) et une garde de bord active. Résultat : plus aucun passage à
  < 0,6 m du bord, mais quelques retraits ne vont pas jusqu'au point de repli (bloqués en sécurité). La tranchée n'est
  qu'un décor plat sous MuJoCo : une chute dedans n'est pas simulée, seulement mesurée par la distance au bord.
- **Carte OSM** : 4/16 seulement. La carte arbre par arbre (15/16) suppose une cartographie préalable fine du terrain,
  irréaliste en général.
- LiDAR 3D : capteur supposé monté retourné (−52°..+7°) et stabilisé, corps du robot exclu par construction, pas de
  végétation basse. Avec le LiDAR 3D, plus d'arrivées à > 5 m (localisation moins précise qu'avec le scan 2D parfaitement segmenté).
- Thermique : modèle à une capacité, portée de détection = **ordre de grandeur** ; seul le comparatif entre profils compte.
- Dépôt : perception du bord de tranchée modélisée (pas simulée) ; dépôt = immobilisation de 8 s (bras non simulés).
- Glissement dans la boue non simulé sous MuJoCo (un seul coefficient de frottement).
- Réglages : les correctifs de navigation ont été trouvés en regardant les graines 100-331 ;
  le jeu 400-431 (84 %) est la mesure de généralisation.

Hypothèses chiffrées : `talos2/params.py` (`SPEC` / `ASSUMPTION`), en-têtes de `osm_map.py`, `thermal.py`,
`delivery.py`, `mujoco_g1/lidar3d.py`.

## Utilisation

```bash
pip install numpy scipy matplotlib pytest mujoco torch pyyaml pillow
python -m pytest -q                                          # 20 tests (~4 min)
bash run_all.sh                                              # tous les chiffres de ce README
python -m talos2.mujoco_g1.run --seed 105 --relief 0.4 --delivery --viewer          # démo G1 (visionneuse MuJoCo)
python -m talos2.mujoco_g1.run --seed 105 --relief 0.4 --delivery --lidar3d         # sans affichage, LiDAR 3D
python -m talos2.mujoco_g1.snapshots --seed 105              # images pour l'IHM
python -m talos2.make_ihm --seeds 105 111                    # IHM -> ihm/talos_ihm.html
```

L'IHM (`ihm/talos_ihm.html`, un seul fichier) montre carte de dessus, tracé, chemin planifié, temps restant, anomalies,
signature thermique, charge et phase de mission, journal, reconstruction 3D, courbes d'erreur et de batterie, et un
panneau « G1 sous MuJoCo » (images + chiffres des benchmarks).

## Structure

```
talos2/params.py         hypothèses (SPEC / ASSUMPTION)
talos2/world.py          monde simulé (vérité terrain), chemins forestiers optionnels
talos2/prior_map.py      carte embarquée arbre par arbre (imparfaite)
talos2/osm_map.py        carte embarquée style OSM + MNT (aucun arbre) + modèle de densité LiDAR
talos2/sensors.py        IMU + odométrie jambes + intégration seule
talos2/localization.py   filtre particulaire (point d'impact / densité / relief)
talos2/planner.py        A* (pente, boue, forêt, chemins, obstacles découverts, zones interdites)
talos2/navigator.py      Navigator indépendant du simulateur
talos2/delivery.py       approche, dépôt, retrait
talos2/thermal.py        signature thermique
talos2/energy.py         bilan énergétique (modèle)
talos2/sim.py            simulateur ponctuel en boucle fermée
talos2/benchmark.py      benchmark sim ponctuel
talos2/mujoco_g1/        pont MuJoCo : scène (MjSpec), politique G1, mission, LiDAR 3D, calibration, benchmark, images
talos2/real_fetch.py     telechargement imagerie / MNT / OSM d'une zone reelle (cache data/real/)
talos2/real_site.py      verite terrain reelle + cartes a niveaux d'information (sat / osm / dem)
talos2/real_benchmark.py benchmark 2 zones x 12 departs x 5 niveaux
talos2/make_ihm.py       IHM (ancienne, fichier unique)
talos_app/               application C2 : datasets, region, route (strategique), mission (temps reel), server, web/, app
run_all.sh               reproduit tous les chiffres
```

## Sources, crédits et licences

- **Politiques pré-entraînées comparées** (non redistribuées, clonées dans `../ext`) :
  - Unitree **unitree_rl_mjlab** (Apache-2.0, © Unitree Robotics) : politique G1 de suivi de vitesse et modèle G1 ;
  - NVIDIA **GR00T-WholeBodyControl** (code Apache-2.0 ; poids sous NVIDIA Open Model License, © NVIDIA) :
    politiques Balance / Walk et modèle `g1_gear_wbc.xml`.

- Code de ce dossier : écrit pour le hackathon (numpy, scipy, matplotlib, mujoco, torch).
- **Unitree Robotics, BSD-3-Clause** (© 2016-2023 HangZhou YuShu TECHNOLOGY CO., LTD.), utilisés sans être copiés ici :
  - `unitree_rl_gym` : modèle `resources/robots/g1_description/g1_12dof.xml` + maillages, politique pré-entraînée
    `deploy/pre_train/g1/motion.pt`, boucle et gains de `deploy/deploy_mujoco/deploy_mujoco.py` + `configs/g1.yaml`
    (réécrits en classe dans `talos2/mujoco_g1/g1_policy.py`).
  - `unitree_mujoco`, `unitree_sdk2_python` : référence pour le passage au robot réel (non utilisés dans ces résultats).
- MuJoCo (Google DeepMind, Apache-2.0).
- **Terrain réel** (téléchargé par `talos2/real_fetch.py` dans `data/real/`, non redistribué, exclu par `.gitignore`) :
  - imagerie **Esri World Imagery** (sources Esri, Maxar, Earthstar Geographics). Conditions d'usage Esri : à faire
    valider par les organisateurs avant toute diffusion publique des images ;
  - MNT **AWS Terrain Tiles** (Mapzen / Linux Foundation ; données SRTM NASA / USGS) ;
  - **© contributeurs OpenStreetMap**, licence ODbL, via l'API Overpass et l'extrait Geofabrik (application C2) ;
  - **ESA WorldCover 2021** (CC-BY 4.0), **Copernicus DEM GLO-30** (© DLR / Airbus, licence Copernicus) ;
  - ligne de contact **DeepStateMap** (deepstatemap.live), polygones de territoire uniquement ;
  - bibliothèques web : **MapLibre GL JS** (BSD-3), **three.js** (MIT), **Chart.js** (MIT) ; tuiles OpenTopoMap (CC-BY-SA), CARTO.
- **Entraînement RL** : MuJoCo Playground (Google DeepMind, Apache-2.0 : environnements, récompenses, randomisation,
  hyperparamètres PPO), Brax (Apache-2.0), JAX (Apache-2.0), MuJoCo / MJX (Apache-2.0).
  Modèles Unitree Go1 et G1 de **MuJoCo Menagerie** (BSD-3-Clause / Apache-2.0 selon le modèle, © Unitree / Google
  DeepMind), copiés dans `rl/models/`.
- Chiffres constructeur : Unitree G1 (~35 kg, batterie 9 000 mAh 13S, ~2 h, charge des bras ~3 kg) ; Livox MID-360
  (champ vertical 59°).
- Le règlement impose de créditer tout élément tiers (une contribution non citée entraîne la disqualification).
