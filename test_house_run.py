import os
import time
import numpy as np
import torch
import mujoco

from run_sim import (
    KPS, KDS, DEFAULT_ANGLES, DOF_POS_SCALE, DOF_VEL_SCALE, ANG_VEL_SCALE,
    ACTION_SCALE, CMD_SCALE, get_gravity_orientation, SIMULATION_DT, CONTROL_DECIMATION,
    LIDAR_ANGLES, LIDAR_MAX_RANGE, NUM_LIDAR_RAYS, NAV_CONFIG
)

xml_path = "unitree_rl_gym/resources/robots/g1_description/scene_tactical_house.xml"
m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)
policy = torch.jit.load("unitree_rl_gym/deploy/pre_train/g1/motion.pt").eval()

goal_pos = np.array([4.5, 0.0])
action = np.zeros(12, dtype=np.float32)
target_dof_pos = DEFAULT_ANGLES.copy()
obs = np.zeros(47, dtype=np.float32)
current_vx = 0.45
geomid_buffer = np.zeros(1, dtype=np.int32)

sim_time = 0.0
total_steps = int(14.0 / SIMULATION_DT)
contacts = 0
falls = 0

wall_geoms = {"house_facade_L", "house_facade_R", "hallway_wall_L", "hallway_wall_R", "stair_wall_L", "stair_wall_R", "landing_wall_back"}
wall_ids = set(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name) for name in wall_geoms if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0)
robot_ids = set(i for i in range(m.ngeom) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i]) != "world")

def scan_lidar_points_relative(m, d, geomid_buffer, yaw):
    points = []
    ranges = []
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    robot_z = d.qpos[2]
    
    for a in LIDAR_ANGLES:
        dists = []
        for (z_rel, ray_pitch) in [(-0.55, 0.0), (-0.30, -0.04), (0.0, 0.0)]:
            ray_dir = np.array([np.cos(yaw + a), np.sin(yaw + a), ray_pitch])
            ray_dir = ray_dir / np.linalg.norm(ray_dir)
            orig = np.array([d.qpos[0] + 0.15 * cos_y, d.qpos[1] + 0.15 * sin_y, robot_z + z_rel])
            dist = mujoco.mj_ray(m, d, orig, ray_dir, None, 1, 1, geomid_buffer)
            if dist >= 0:
                hit_pt = orig + dist * ray_dir
                # Check height relative to robot base and filter walkable surfaces
                gid = geomid_buffer[0]
                gname = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) if gid >= 0 else "") or ""
                walkable = any(k in gname for k in ["floor", "stair_step", "stair_lip", "landing", "target", "extraction", "lawn"])
                if not walkable and hit_pt[2] > (robot_z - 0.65):
                    dists.append(dist)
        min_d = min(dists) if dists else LIDAR_MAX_RANGE
        ranges.append(min_d)
        if dists:
            px = min_d * np.cos(a)
            py = min_d * np.sin(a)
            points.append((px, py, min_d, a))
    return np.array(ranges), points

def vfh_nav(m, d, goal_pos, geomid_buffer, mission_done=False):
    if mission_done:
        return 0.0, 0.0, 0.0, "CIBLE ATTEINTE (EN GARDE)", 0.0, np.ones(NUM_LIDAR_RAYS) * LIDAR_MAX_RANGE

    robot_pos = d.qpos[:2]
    to_goal = goal_pos - robot_pos
    dist_goal = np.linalg.norm(to_goal)

    if dist_goal <= 0.28:
        return 0.0, 0.0, 0.0, "CIBLE ATTEINTE (EN GARDE)", 0.0, np.ones(NUM_LIDAR_RAYS) * LIDAR_MAX_RANGE

    quat = d.qpos[3:7]
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    goal_angle_world = np.arctan2(to_goal[1], to_goal[0])
    goal_angle_local = (goal_angle_world - yaw + np.pi) % (2.0 * np.pi) - np.pi

    ranges, points = scan_lidar_points_relative(m, d, geomid_buffer, yaw)

    R_robot = 0.32
    lookahead = 2.0
    best_angle = 0.0
    min_cost = 999.0

    for a in LIDAR_ANGLES:
        cos_a = np.cos(a)
        sin_a = np.sin(a)
        blocked = False
        min_obs_dist = lookahead
        for px, py, d_orig, ray_a in points:
            d_par = px * cos_a + py * sin_a
            d_perp = abs(-px * sin_a + py * cos_a)
            if 0 < d_par < lookahead and d_perp < R_robot:
                blocked = True
                break
            if 0 < d_par < lookahead:
                min_obs_dist = min(min_obs_dist, d_orig)

        if not blocked:
            cost = abs(a - goal_angle_local) + 0.30 * abs(a) + 0.30 * (lookahead - min_obs_dist)
            if cost < min_cost:
                min_cost = cost
                best_angle = a

    if min_cost > 900.0 and points:
        best_angle = max(points, key=lambda p: p[2])[3]

    min_front_obs = min([p[2] for p in points if abs(p[3]) < np.radians(14)] + [LIDAR_MAX_RANGE])
    if min_front_obs < 0.36:
        desired_vx = 0.0
    else:
        obstacle_slowdown = min(1.0, max(0.18, (min_front_obs - 0.36) / 0.60))
        speed_scale = min(1.0, max(0.35, dist_goal / 0.8)) * obstacle_slowdown
        max_vx = NAV_CONFIG.get("max_vx", 0.52)
        desired_vx = float(np.clip(max_vx * max(0.15, np.cos(best_angle)) * speed_scale, 0.12, max_vx))

    desired_vy = float(np.clip(np.sin(best_angle) * 0.25, -0.35, 0.35))
    desired_vyaw = float(np.clip(best_angle * 1.5, -0.65, 0.65))
    phase = f"VFH AUTONOME [CAP {np.degrees(best_angle):+4.1f}°]"

    return desired_vx, desired_vy, desired_vyaw, phase, best_angle, ranges

mission_done = False

print("=" * 70)
print("TEST RUN: G1 ENTERING REALISTIC HOUSE AND CLIMBING STAIRS")
print("=" * 70)

for s in range(total_steps):
    tau = (target_dof_pos - d.qpos[7:]) * KPS - d.qvel[6:] * KDS
    d.ctrl[:] = tau
    mujoco.mj_step(m, d)
    sim_time += SIMULATION_DT
    
    # Check contacts with walls
    for i in range(d.ncon):
        con = d.contact[i]
        if (con.geom1 in wall_ids and con.geom2 in robot_ids) or (con.geom2 in wall_ids and con.geom1 in robot_ids):
            contacts += 1
            
    # Check fall
    if d.qpos[2] < 0.35:
        falls += 1
        print(f"FALL AT t={sim_time:.2f}s! X={d.qpos[0]:.2f}, Z={d.qpos[2]:.2f}")
        break
        
    dist_goal = np.linalg.norm(goal_pos - d.qpos[:2])
    if not mission_done and dist_goal <= 0.28:
        mission_done = True
        print(f"\n--> MISSION SUCCESS! REACHED 1ST FLOOR TARGET AT t={sim_time:.2f}s! (X={d.qpos[0]:.2f}m, Z={d.qpos[2]:.2f}m)\n")
        
    if s % CONTROL_DECIMATION == 0:
        desired_vx, desired_vy, desired_vyaw, phase, best_angle, ranges = vfh_nav(
            m, d, goal_pos, geomid_buffer, mission_done
        )
        
        current_vx = np.clip(desired_vx, current_vx - 0.015, current_vx + 0.015)
        cmd = np.array([current_vx, desired_vy, desired_vyaw], dtype=np.float32)
        
        qj = (d.qpos[7:] - DEFAULT_ANGLES) * DOF_POS_SCALE
        dqj = d.qvel[6:] * DOF_VEL_SCALE
        quat = d.qpos[3:7]
        omega = d.qvel[3:6] * ANG_VEL_SCALE
        grav = get_gravity_orientation(quat)
        
        period = 0.8
        phase_clock = (sim_time % period) / period
        obs[:3] = omega
        obs[3:6] = grav
        obs[6:9] = cmd * CMD_SCALE
        obs[9:21] = qj
        obs[21:33] = dqj
        obs[33:45] = action
        obs[45:47] = np.array([np.sin(2*np.pi*phase_clock), np.cos(2*np.pi*phase_clock)], dtype=np.float32)
        
        with torch.no_grad():
            action = policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
            
        if mission_done:
            target_dof_pos = DEFAULT_ANGLES.copy()
        else:
            target_dof_pos = action * ACTION_SCALE + DEFAULT_ANGLES
            
            # Anti-fall & stair climb stumble reflex
            pitch_err = grav[0]
            roll_err = grav[1]
            if abs(pitch_err) > 0.035:
                target_dof_pos[0] -= pitch_err * 0.35
                target_dof_pos[4] -= pitch_err * 0.25
                target_dof_pos[6] -= pitch_err * 0.35
                target_dof_pos[10] -= pitch_err * 0.25
            if abs(roll_err) > 0.035:
                target_dof_pos[1] += roll_err * 0.28
                target_dof_pos[5] += roll_err * 0.16
                target_dof_pos[7] += roll_err * 0.28
                target_dof_pos[11] += roll_err * 0.16

    if s % 250 == 0:
        zone = "EXTERIOR" if d.qpos[0] < 1.35 else ("HALLWAY" if d.qpos[0] < 2.35 else ("STAIRS" if d.qpos[0] < 3.65 else "MEZZANINE"))
        print(f"t={sim_time:4.1f}s | ZONE: {zone:<9} | POS: X={d.qpos[0]:.2f}m, Y={d.qpos[1]:.2f}m, Z={d.qpos[2]:.2f}m | Pitch={np.degrees(-grav[0]):+4.1f}° | Contacts={contacts}")

print(f"\nFinal Result: X={d.qpos[0]:.2f}m, Z={d.qpos[2]:.2f}m, Falls={falls}, Wall Contacts={contacts}")
