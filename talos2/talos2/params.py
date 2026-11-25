"""Tous les parametres du modele, avec leur statut.

SPEC        = donnee constructeur ou du cahier des charges (source indiquee)
ASSUMPTION  = hypothese de modelisation, a remplacer par une mesure (MuJoCo / robot reel)
Rien ici n'est presente comme un resultat : les resultats sortent de benchmark.py.
"""
from dataclasses import dataclass


@dataclass
class Params:
    # --- Robot (Unitree G1) ---
    robot_radius: float = 0.35          # ASSUMPTION encombrement lateral bras compris
    v_cmd_max: float = 1.0              # ASSUMPTION vitesse de croisiere prudente (SPEC: max ~2 m/s)
    w_cmd_max: float = 0.8              # ASSUMPTION rad/s
    v_back: float = -0.25               # ASSUMPTION m/s, marche arriere de degagement
    payload_kg: float = 3.0             # SPEC: charge max. des bras ~3 kg (Generation Robots)
    mass_kg: float = 35.0               # SPEC: ~35 kg avec batterie (Unitree)

    # --- Batterie ---
    battery_wh: float = 421.0           # SPEC: 13S Li-ion 9000 mAh (13 x 3.6 V x 9 Ah)
    battery_reserve: float = 0.15       # ASSUMPTION reserve de securite (retour / imprevu)

    # --- Terrain reel (monde simule) ---
    mud_slip: float = 0.85              # ASSUMPTION fraction de vitesse conservee dans la boue (15 % de glissement)
    slope_slow_from: float = 0.15       # ASSUMPTION pente (m/m) a partir de laquelle on ralentit
    obstacle_contact_pause: float = 0.6 # ASSUMPTION s d'arret apres un contact

    # --- Capteurs ---
    lidar_rays: int = 72                # SPEC: LiDAR 3D 360 deg (simule ici en coupe 2D horizontale)
    lidar_range: float = 25.0           # ASSUMPTION m
    lidar_sigma: float = 0.05           # ASSUMPTION m, bruit de mesure
    lidar_dropout: float = 0.03         # ASSUMPTION fraction de rayons perdus
    scan_dt: float = 0.2                # ASSUMPTION 5 Hz
    gyro_noise: float = 0.0003          # ASSUMPTION rad/s par echantillon a 10 Hz (~0.005 deg/s/rtHz, MEMS courant)
    gyro_bias_sd: float = 0.0003        # ASSUMPTION rad/s (~1 deg/min de biais residuel apres calibration)
    gyro_bias_walk: float = 0.00002     # ASSUMPTION rad/s/sqrt(s)
    odom_scale_err: float = 0.03        # ASSUMPTION erreur d'echelle odometrie jambes
    odom_noise: float = 0.05            # ASSUMPTION m/s

    # --- Carte a priori (memoire) ---
    prior_keep_trees: float = 0.70      # ASSUMPTION part des arbres presents dans la carte
    prior_pos_sigma: float = 0.40       # ASSUMPTION m, erreur de position des objets cartes
    prior_ghost_trees: float = 0.05     # ASSUMPTION faux objets (carte perimee)
    prior_res: float = 0.2              # m

    # --- Filtre a particules ---
    n_particles: int = 6000
    pf_sigma_hit: float = 0.5           # m, tolere l'erreur de carte (0,4 m) et la quantification
    pf_w_rand: float = 0.35             # plancher uniforme (objets non cartes, anomalies)
    pf_temper: float = 0.25             # attenuation (rayons correles)
    pf_rays_used: int = 48
    pf_w_hit: float = 0.65
    pf_relief_sigma: float = 0.07       # ASSUMPTION erreur combinee mesure de pente + MNT (m/m)
    pf_relief_temper: float = 0.3
    imu_slope_noise: float = 0.02       # ASSUMPTION bruit de l'inclinaison deduite de la centrale inertielle (m/m)
    pf_mode: str = "endpoint"            # 'beam' (lancer de rayons) ou 'endpoint' (champ de vraisemblance)
    pf_unseen_lik: float = 0.12         # vraisemblance d'un arbre cartographie non vu par le LiDAR
    pf_odom_gate: bool = False          # garde de coherence recalage / odometrie (application terrain reel)
    pf_lam_calib: bool = False          # auto-etalonnage de la densite LiDAR (cartes terrain reel)
    pf_inject_sd_max: float = 15.0      # m, ecart-type max des particules d injection (MCL augmente)

    # --- Evitement local ---
    corridor_half: float = 0.55         # m, demi-largeur du couloir de balayage (rayon robot + marge)
    stop_dist: float = 0.8              # m, arret si obstacle dans le couloir
    slow_dist: float = 2.0              # m, ralentissement progressif
    close_dist: float = 0.5             # m, frolement
    v_slow_min: float = 0.0             # m/s plancher dans la zone de ralentissement (G1 : ~0 m/s sous 0,15 m/s en pente)
    slope_speed_limit: bool = False     # reduire la vitesse selon la pente mesuree (IMU) ; active sous MuJoCo
    progress_watchdog: float = 25.0     # s sans reduire le chemin restant de 1 m -> degagement + replanification (0 = off)
    v_escape: float = 0.3               # m/s, degagement vers l'ouverture quand l'arriere est bloque
    stall_detect: bool = True           # consigne > 0 mais odometrie ~0 pendant ~3 s -> compte comme bloque

    # --- Planification ---
    plan_inflation: float = 1.0         # m autour des obstacles connus
    plan_slope_max: float = 0.0         # m/m, au-dela : penalite plan_slope_penalty (0 = aucune) ; MuJoCo/G1 : voir mujoco_params
    plan_slope_penalty: float = 150.0   # cout supplementaire par m/m au-dessus de plan_slope_max
    plan_start_escape_cost: float = 20.0  # cout d'une cellule bloquee pres du depart (on peut en sortir, pas y passer)
    plan_border: float = 3.0            # m interdits le long du bord de la carte (bord du MNT)
    plan_mud_cost: float = 1.6
    plan_slope_cost: float = 6.0
    anomaly_min_points: int = 4
    map_hits_as_obstacles: bool = False # carte sans arbres (OSM) : chaque impact LiDAR -> obstacle local
    hit_obstacle_radius: float = 1.0    # m

    # --- Mission ---
    goal_tolerance_est: float = 1.5     # m (sur la position estimee)
    success_tolerance_true: float = 5.0 # m (verite terrain) ; les derniers metres relevent de la perception locale de la tranchee (non simulee)
    max_time: float = 1800.0            # s
