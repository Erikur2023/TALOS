# TALOS - Track 1 : Navigation Autonome & Mobilité en Terrain Hostile
**Équipe #225 - ENSTA Campus de Brest (Tutelle DGA)**  
**Hackathon Robot Humanoïde Défense - Grand Palais 2026**

---

## 1. Vue d'Ensemble du Socle de Simulation

Ce banc de simulation implémente l'architecture matérielle et logicielle officielle validée par **Innov8 & Unitree** pour le robot humanoïde **Unitree G1 (29 DDL)** et le quadrupède **Go2**.

```
TALOS/
├── control/
│   └── g1_controller.py       # Contrôleur PD, postures tactiques (STAND, READY, POINT, DAMP)
├── navigation/
│   ├── odometry_estimator.py  # Simulateur Point-LIO (fusion LiDAR 3D + IMU 100 Hz)
│   └── traversability_grid.py # Grille 2.5D d'élévation & détection des dévers (> 25°)
├── safety/
│   └── talos_watchdog.py      # Watchdog IMU temps-réel (Coupure amortie si tilt > 25°)
├── simulation/
│   ├── scene_talos_defense.xml# Scénario tactique : Sas 70cm, décombres, caisse cible
│   └── talos_simulator.py     # Boucle MuJoCo temps-réel avec télémétrie & visualiseur
└── run_sim.py                 # Point d'entrée unique
```

---

## 2. Piliers Techniques TALOS Implémentés

### Pilier 1 : Sûreté Déterministe & Watchdog IMU (`safety/talos_watchdog.py`)
- Surveillance haute fréquence des angles d'Euler (Roll / Pitch) issus du quaternion IMU (`imu_quat`).
- **Seuil d'alerte** : $|\text{Roll}|$ ou $|\text{Pitch}| > 18^\circ$.
- **Seuil critique d'arrêt d'urgence** : $|\text{Roll}|$ ou $|\text{Pitch}| \ge 25^\circ$.
- **Action de sécurité** : Basculement instantané en mode `EMERGENCY_DAMP` ($K_p = 0$, $K_d = 8.0\,\text{N}\cdot\text{s/m}$) pour dissiper l'énergie sans forcer sur les réducteurs mécaniques.

### Pilier 2 : Odométrie Laser-Inertielle (`navigation/odometry_estimator.py`)
- Modélisation du pipeline **Point-LIO** filtrant les vibrations et impacts au sol de la démarche bipède.
- Fourniture de la pose 6-DOF $(x, y, z, \phi, \theta, \psi)$ et de la distance cumulée.

### Pilier 3 : Traversabilité 2.5D (`navigation/traversability_grid.py`)
- Extraction locale du gradient de pente pour détecter les gravats et adapter la garde au sol.

### Pilier 4 : Scénario Tactique Grand Palais (`simulation/scene_talos_defense.xml`)
- Intégration du modèle officiel Unitree G1 29 DDL.
- **Zone 1** : Sas de franchissement calibré à **70 cm** (critère strict DGA).
- **Zone 2** : Zone de décombres et de dévers (pentes jusqu'à 25°).
- **Zone 3** : Cible d'inspection tactique (caisse avec marqueur de visée).

---

## 3. Lancement et Commandes Interactives

### Lancement avec Visualiseur 3D interactif :
```bash
python run_sim.py
```

### Commandes clavier dans la fenêtre MuJoCo :
* **`1`** : Posture Debout Nominale (`STAND`)
* **`2`** : Garde Tactique (`TACTICAL_READY` - mains levées à mi-hauteur)
* **`3`** : Désignation d'Objectif (`POINT_TARGET` - bras droit tendu vers la cible)
* **`4`** : Injection d'une perturbation sévère (Test du déclenchement Watchdog)
* **`R`** : Réinitialiser la simulation et la pose du robot
* **`Espace`** : Pause / Reprise

### Test rapide en mode console (Headless) :
```bash
python run_sim.py --headless
```
