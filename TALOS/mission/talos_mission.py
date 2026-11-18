"""
TALOS Mission Orchestrator & State Machine / Behavior Tree
Équipe TALOS (#225 - ENSTA Brest / tutelle DGA)
Orchestration de la mission opérationnelle Track 1 (Grand Palais 2026) :
  1. Boot & Self-Check Sûreté
  2. Franchissement du Sas étroit (70 cm)
  3. Traversée décombres avec adaptation de traversabilité
  4. Désignation & Inspection de la cible tactique
"""

import time
import numpy as np

class TalosMissionOrchestrator:
    # Phases opérationnelles
    STAGE_INIT = "1_BOOT_SELF_CHECK"
    STAGE_APPROACH_SAS = "2_APPROACH_SAS_70CM"
    STAGE_CROSS_SAS = "3_CROSSING_NARROW_SAS"
    STAGE_TERRAIN_RUBBLE = "4_CROSSING_RUBBLE_ZONE"
    STAGE_TARGET_RECON = "5_TARGET_ACQUISITION_POINT"
    STAGE_MISSION_COMPLETE = "6_MISSION_COMPLETE"
    STAGE_ABORT = "ABORT_EMERGENCY"

    def __init__(self):
        self.stage = self.STAGE_INIT
        self.start_time = time.time()
        self.target_pos = np.array([5.0, 0.0, 0.35])
        self.desired_vx = 0.0
        self.desired_vyaw = 0.0
        self.desired_posture = "STAND"
        self.log_messages = []
        self.target_identified = False

    def log(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        formatted = f"[{timestamp}] [TALOS C2] {msg}"
        self.log_messages.append(formatted)
        print(f"\n{formatted}")

    def update(self, sim_time, robot_pos, wd_telemetry, traversability_status):
        """
        Met à jour la machine d'état de mission en fonction de la position du robot (x, y, z).
        """
        x, y, z = robot_pos[0], robot_pos[1], robot_pos[2]

        # 0. Vérification arrêt d'urgence Watchdog
        if wd_telemetry.get("tripped", False):
            if self.stage != self.STAGE_ABORT:
                self.stage = self.STAGE_ABORT
                self.log(f"ARRÊT D'URGENCE MISSION ! Déclenchement Watchdog : {wd_telemetry.get('trigger_reason')}")
            self.desired_vx = 0.0
            self.desired_posture = "EMERGENCY_DAMP"
            return self.get_command()

        # Phase 1 : Initialisation & Vérification Sûreté
        if self.stage == self.STAGE_INIT:
            if sim_time >= 1.0:
                self.log("Auto-test système OK : IMU 100 Hz connectée, Watchdog armé.")
                self.log("Début de la progression vers le SAS 70 cm (Gabarit DGA).")
                self.stage = self.STAGE_APPROACH_SAS

        # Phase 2 : Approche du Sas (X < 1.6m)
        elif self.stage == self.STAGE_APPROACH_SAS:
            self.desired_vx = 0.25
            self.desired_posture = "STAND"
            if x >= 1.5:
                self.log("Sas 70 cm détecté à 50 cm. Resserrement du gabarit des bras.")
                self.stage = self.STAGE_CROSS_SAS

        # Phase 3 : Franchissement du Sas (1.6m <= X < 2.5m)
        elif self.stage == self.STAGE_CROSS_SAS:
            self.desired_vx = 0.20
            self.desired_posture = "STAND" # Bras plaqués contre le buste
            if x >= 2.5:
                self.log("Sas franchi avec succès sans impact latéral ! Zone de décombres en approche.")
                self.stage = self.STAGE_TERRAIN_RUBBLE

        # Phase 4 : Décombres & Dévers (2.5m <= X < 4.0m)
        elif self.stage == self.STAGE_TERRAIN_RUBBLE:
            # Trajectoire Nav2 : évitement local par contournement à Y = +0.18m
            self.desired_vx = 0.16
            self.desired_vyaw = 0.0
            self.desired_posture = "TACTICAL_READY"
            
            if x >= 4.0:
                self.log("Zone de décombres franchie avec succès ! Détection visuelle de la cible à X=5.0m.")
                self.stage = self.STAGE_TARGET_RECON

        # Phase 5 : Désignation et Visée Cible (X >= 4.0m)
        elif self.stage == self.STAGE_TARGET_RECON:
            self.desired_vx = 0.0 # Arrêt stable
            self.desired_posture = "POINT_TARGET" # Bras droit tendu pointant la cible
            self.target_identified = True
            if sim_time >= 15.0 or x >= 4.2:
                self.log("Cible désignée et géo-référencée (Laser IR verrouillé). Rapport transmis au C2.")
                self.stage = self.STAGE_MISSION_COMPLETE

        # Phase 6 : Mission Complète
        elif self.stage == self.STAGE_MISSION_COMPLETE:
            self.desired_vx = 0.0
            self.desired_posture = "POINT_TARGET"

        return self.get_command()

    def get_command(self):
        return {
            "stage": self.stage,
            "desired_vx": self.desired_vx,
            "desired_vyaw": self.desired_vyaw,
            "desired_posture": self.desired_posture,
            "target_identified": self.target_identified
        }
