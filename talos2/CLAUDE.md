# TALOS v2 — contexte de passation (lu automatiquement par Claude Code)

## Projet
Hackathon « Humanoid Defense Robot » (Artefact Adopt AI + AID), Grand Palais, Paris, 2–4 déc. 2026. Track 1, barrière B.
Scénario : un robot (Unitree G1) traverse en autonomie forêt / boue / relief, SANS GNSS (brouillage), pour ravitailler
des soldats abrités dans une tranchée. Il se localise par IMU + odométrie de jambes + recalage LiDAR sur une carte embarquée
imparfaite, replanifie autour des obstacles et affiche tout sur une IHM.

## Règlement (à respecter)
- Validation sous MuJoCo obligatoire avant tout accès au robot réel (unitree_mujoco, DDS, CycloneDDS 0.10.2, unitree_sdk2_python). Windows = MuJoCo uniquement.
- Tout contenu tiers doit être crédité (sinon disqualification). Dépôts Unitree = BSD-3 : réutilisables avec crédit (voir README §Crédits).
- Résultats publics ; présence sur place les 3 jours.

## État actuel (talos2/, Python pur numpy, testé)
- params.py (hypothèses marquées SPEC/ASSUMPTION), world.py (forêt/relief/boue synthétiques), prior_map.py (carte embarquée
  imparfaite : 30 % d'arbres manquants, décalage 0,4 m, faux objets), sensors.py, localization.py (filtre particulaire 6 états,
  modèle endpoint + recalage relief), planner.py (A*), energy.py (calé sur ~2 h constructeur, 421 Wh), sim.py (boucle fermée,
  robot = point avec glissement), benchmark.py, plots.py, make_ihm.py → ihm/talos_ihm.html. tests/ : `pytest` (6 tests).
- Benchmark graines 100–115 : recalage 15/16 réussites, erreur médiane 0,67 m, 0 contact ; odométrie seule 0/16, 8,9 m.
- Ancien projet (racine Hackathon/) = audité, ne pas réutiliser ses chiffres (constantes codées en dur, faux benchmark).

## Ressources déjà sur le PC
- ../unitree_rl_gym/deploy/deploy_mujoco/deploy_mujoco.py + configs/g1.yaml + deploy/pre_train/g1/motion.pt
  (politique de marche G1 12 ddl, obs 47, consigne cmd=[vx, vy, wz], cmd_scale [2,2,0.25], 50 Hz, dt 0.002, décimation 10).
- ../unitree_rl_gym/resources/robots/g1_description/g1_12dof.xml + scene.xml
- ../unitree_mujoco, ../unitree_sdk2_python. Python 3.12 + mujoco + torch déjà utilisés (voir ../run_sim.py).

## Plan à exécuter (dans cet ordre, sans s'arrêter pour demander ; noter chaque choix dans NOTES_NUIT.md)
1. Extraire de sim.py un `Navigator` (filtre particulaire + planif + évitement + ETA + détection d'anomalies) indépendant du
   simulateur : entrée = gyro, odométrie, scan LiDAR, pente IMU ; sortie = consigne (vx, wz). Le sim point existant doit
   donner les mêmes résultats (relancer benchmark).
2. Pont MuJoCo : talos2/mujoco_g1/ — générer une scène MJCF depuis World (hfield relief + cylindres d'arbres près du couloir
   + tranchée), inclure g1_12dof.xml, boucle = deploy_mujoco.py avec cmd venant du Navigator ; IMU depuis capteurs / qvel,
   odométrie bruitée, LiDAR via mujoco.mj_ray (ou mj_multiRay) en excluant le robot. Mode sans affichage pour benchmark +
   mode viewer pour la démo. Mesurer : chutes, réussite, erreur de localisation. Respecter le domaine de consigne de la
   politique (vx ≲ 0,5–0,8 m/s, vérifier empiriquement).
3. Corriger le scénario 111 (piège de planification locale : poche d'arbres).
4. Carte « réelle » : remplacer la carte arbre-par-arbre par ce qu'une vraie carte fournit (MNT/relief, lisières forêt,
   chemins, type de sol, style OSM) et recaler sur ces structures ; mesurer la dégradation honnêtement.
5. LiDAR 3D (G1 : Livox MID-360) : nuage multi-couches, segmentation sol/obstacles, pente locale.
6. Signature thermique (modèle simple, hypothèses marquées) + phase de dépôt du ravitaillement (charge ≤ 3 kg bras,
   approche à distance de sécurité de la tranchée, dépôt, retrait).
7. Relancer benchmark, mettre à jour IHM (panneaux thermique / charge, vue MuJoCo si possible), README (résultats, limites,
   crédits), tests. Ne jamais publier un chiffre qui ne sort pas d'un run reproductible.
