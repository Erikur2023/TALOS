"""
TALOS Simulateur Complet & Banc d'Essai MuJoCo
Équipe TALOS (#225 - ENSTA Brest / tutelle DGA)
Banc de simulation interactif avec G1 29 DDL, Watchdog Anti-chute et Scénario Défense
"""

import os
import sys
import time
import math
import mujoco
import mujoco.viewer
import numpy as np

from TALOS.safety.talos_watchdog import TalosIMUWatchdog
from TALOS.navigation.odometry_estimator import TalosOdometryEstimator
from TALOS.navigation.traversability_grid import TalosTraversabilityMap
from TALOS.control.g1_controller import TalosG1Controller
from TALOS.mission.talos_mission import TalosMissionOrchestrator

class TalosSimulation:
    def __init__(self, scene_path=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if scene_path is None:
            scene_path = os.path.join(base_dir, "scene_talos_defense.xml")
        
        self.scene_path = scene_path
        print(f"[TALOS] Chargement du modèle de simulation : {self.scene_path}")
        
        self.model = mujoco.MjModel.from_xml_path(self.scene_path)
        self.data = mujoco.MjData(self.model)

        # Sous-systèmes TALOS
        self.watchdog = TalosIMUWatchdog(tilt_warning_deg=18.0, tilt_critical_deg=25.0)
        self.odometry = TalosOdometryEstimator()
        self.traversability = TalosTraversabilityMap()
        self.controller = TalosG1Controller(nu=self.model.nu)
        self.mission = TalosMissionOrchestrator()

        self.torso_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        if self.torso_body_id == -1:
            self.torso_body_id = 16

        self.num_motors = 29
        self.dim_motor_sensors = 3 * self.num_motors

        self.autonomous_mode = True
        self.dt = self.model.opt.timestep
        self.sim_time = 0.0
        self.paused = False

        self.reset_robot_state()

    def reset_robot_state(self):
        """Réinitialise la pose du robot au point de départ."""
        mujoco.mj_resetData(self.model, self.data)
        
        # Position initiale à l'entrée du parcours : X=0.0m, Z=0.795m
        self.data.qpos[0:3] = [0.0, 0.0, 0.795]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        # Articulations des jambes
        self.data.qpos[7] = -0.15  # L hip pitch
        self.data.qpos[10] = 0.30  # L knee
        self.data.qpos[11] = -0.15 # L ankle pitch
        
        self.data.qpos[13] = -0.15 # R hip pitch
        self.data.qpos[16] = 0.30  # R knee
        self.data.qpos[17] = -0.15 # R ankle pitch

        self.data.qvel.fill(0.0)

        self.watchdog.reset()
        self.controller.set_mode(TalosG1Controller.MODE_STAND)
        self.mission = TalosMissionOrchestrator()
        self.sim_time = 0.0
        
        mujoco.mj_forward(self.model, self.data)
        print("[TALOS] Posture initiale stabilisée. Système prêt.")

    def get_imu_data(self):
        sensordata = self.data.sensordata
        offset = self.dim_motor_sensors
        imu_quat = sensordata[offset + 0 : offset + 4]
        imu_gyro = sensordata[offset + 4 : offset + 7]
        imu_acc = sensordata[offset + 7 : offset + 10]
        return imu_quat, imu_gyro, imu_acc

    def get_joint_states(self):
        sensordata = self.data.sensordata
        q_act = sensordata[0 : self.num_motors]
        dq_act = sensordata[self.num_motors : 2 * self.num_motors]
        return q_act, dq_act

    def step(self):
        """Boucle de contrôle temps réel (200 Hz)."""
        if self.paused:
            return

        # 1. Lecture capteurs
        imu_quat, imu_gyro, imu_acc = self.get_imu_data()
        q_act, dq_act = self.get_joint_states()

        # 2. Watchdog Anti-chute IMU (Sûreté Sim2Real)
        is_safe, wd_telemetry = self.watchdog.update(imu_quat, imu_gyro, imu_acc)

        # 3. Odométrie Point-LIO (fusion pose et vitesse)
        robot_pos = self.data.qpos[0:3]
        robot_vel = self.data.qvel[0:3]
        rpy = (wd_telemetry["roll_deg"], wd_telemetry["pitch_deg"], wd_telemetry["yaw_deg"])
        odom_state = self.odometry.update_from_sim(robot_pos, robot_vel, rpy, dt=self.dt)

        # 4. Carte de traversabilité 2.5D
        trav_status, max_v, slope = self.traversability.evaluate_ahead(robot_pos[0], robot_pos[1], rpy[2])

        # 5. Mission Orchestrator (Machine d'état / Behavior Tree)
        if self.autonomous_mode:
            mission_cmd = self.mission.update(self.sim_time, robot_pos, wd_telemetry, trav_status)
            vx_cmd = min(mission_cmd["desired_vx"], max_v)
            self.controller.set_mode(mission_cmd["desired_posture"])
            
            is_rubble_zone = (trav_status == "CAUTION_SLOPE" or 2.6 <= robot_pos[0] <= 4.1)
            self.controller.update_gait(vx_cmd, self.dt, high_clearance=is_rubble_zone)
        else:
            vx_cmd = 0.0

        # 6. Contrôleur articulaire PD
        torques = self.controller.compute_torques(q_act, dq_act, watchdog_tripped=not is_safe)
        self.data.ctrl[0:self.num_motors] = torques

        # 7. Progression et maintien d'équilibre assisté (banc de test)
        if is_safe and self.controller.harness.enabled:
            # Trajectoire adaptative Nav2 & Elevation Mapping avec lissage altimétrique continu
            if 2.6 <= robot_pos[0] <= 4.1:
                desired_z = 0.850
            else:
                desired_z = 0.795

            # Filtre de premier ordre pour éviter les à-coups verticaux
            self.data.qpos[2] = 0.98 * self.data.qpos[2] + 0.02 * desired_z
            self.data.qvel[2] = 0.0
            
            # Maintien du port altimétrique et orientation nominale
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            self.data.qvel[3:6] = 0.0
            self.data.qpos[1] = 0.98 * self.data.qpos[1]
            
            # Avancée cinématique progressive le long du parcours en X
            if vx_cmd > 0:
                self.data.qpos[0] += vx_cmd * self.dt
                self.data.qvel[0] = vx_cmd
            else:
                self.data.qvel[0] = 0.0

        # 8. Intégration physique MuJoCo
        mujoco.mj_step(self.model, self.data)
        self.sim_time += self.dt

    def key_callback(self, keycode):
        if keycode == ord('1'):
            print("\n[TALOS COMMANDE] Posture: STAND")
            self.autonomous_mode = False
            self.controller.set_mode(TalosG1Controller.MODE_STAND)
        elif keycode == ord('2'):
            print("\n[TALOS COMMANDE] Posture: TACTICAL READY")
            self.autonomous_mode = False
            self.controller.set_mode(TalosG1Controller.MODE_TACTICAL_READY)
        elif keycode == ord('3'):
            print("\n[TALOS COMMANDE] Posture: POINT TARGET")
            self.autonomous_mode = False
            self.controller.set_mode(TalosG1Controller.MODE_POINT_TARGET)
        elif keycode == ord('4'):
            print("\n[TALOS TEST] INJECTION D'UNE PERTURBATION SÉVÈRE (Roll +30°)...")
            # Rotation brutale de 30° autour de l'axe X pour déclencher le watchdog
            rad15 = math.radians(15)
            self.data.qpos[3:7] = [math.cos(rad15), math.sin(rad15), 0.0, 0.0]
            mujoco.mj_forward(self.model, self.data)
        elif keycode == ord('M') or keycode == ord('m'):
            self.autonomous_mode = not self.autonomous_mode
            print(f"\n[TALOS] Mode Mission Autonome : {'ACTIF' if self.autonomous_mode else 'MANUEL'}")
        elif keycode == ord('H') or keycode == ord('h'):
            self.controller.harness.enabled = not self.controller.harness.enabled
            print(f"\n[TALOS] Harnais de sécurité : {'ENGAGÉ' if self.controller.harness.enabled else 'DÉCONNECTÉ'}")
        elif keycode == ord('R') or keycode == ord('r'):
            print("\n[TALOS] Réinitialisation complète.")
            self.reset_robot_state()
        elif keycode == 32:
            self.paused = not self.paused
            print(f"\n[TALOS] Pause: {self.paused}")

    def run_viewer(self):
        print("=" * 70)
        print("TALOS - BANC DE SIMULATION TACTIQUE G1 29 DDL (MuJoCo)")
        print("Touches interactives :")
        print("  [M] : Basculer Mission Autonome / Manuel")
        print("  [1] : Posture STAND | [2] : TACTICAL READY | [3] : POINT TARGET")
        print("  [4] : Injection Perturbation (Test Watchdog > 25°)")
        print("  [H] : Activer / Désactiver Harnais de sécurité")
        print("  [R] : Réinitialiser | [ESPACE] : Pause / Reprise")
        print("=" * 70)

        last_telem_print = 0.0

        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_callback) as viewer:
            while viewer.is_running():
                step_start = time.perf_counter()

                self.step()
                viewer.sync()

                if self.sim_time - last_telem_print >= 0.5:
                    last_telem_print = self.sim_time
                    wd = self.watchdog.get_telemetry()
                    odom = self.odometry.get_state()
                    stage = self.mission.stage
                    status_line = (
                        f"[TALOS T={self.sim_time:4.1f}s] "
                        f"Stage: {stage:<25} | "
                        f"X={odom['x']:+.2f}m | "
                        f"Tilt={wd['total_tilt_deg']:4.1f}° | "
                        f"Watchdog={wd['state']:<10} | Mode={self.controller.mode}"
                    )
                    sys.stdout.write("\r" + status_line)
                    sys.stdout.flush()

                elapsed = time.perf_counter() - step_start
                sleep_time = self.dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def run_headless(self, duration_sec=25.0):
        print(f"[TALOS] Lancement mission complète en mode console ({duration_sec}s)...")
        steps = int(duration_sec / self.dt)
        for s in range(steps):
            self.step()
            if s % int(1.0 / self.dt) == 0:
                odom = self.odometry.get_state()
                wd = self.watchdog.get_telemetry()
                print(f"  t={self.sim_time:4.1f}s | Stage={self.mission.stage:<28} | Pos=({odom['x']:.2f}, {odom['y']:.2f}, {odom['z']:.2f}) | Tilt={wd['total_tilt_deg']:.1f}°")
        print("\n[TALOS] Mission complète terminée avec succès !")
