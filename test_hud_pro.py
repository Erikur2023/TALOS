import cv2
import numpy as np

def draw_pro_tactical_hud(frame_bgr, ranges, best_angle, yaw_deg, pitch_deg, roll_deg, dist_to_goal, speed_ms, sim_time):
    h, w = frame_bgr.shape[:2]
    cx, cy = w // 2, h // 2

    # Couleurs militaires modernes (BGR)
    CYAN = (248, 189, 56)      # #38bdf8
    GREEN = (80, 220, 100)     # Vert tactique
    AMBER = (40, 180, 255)     # Ambre alerte
    WHITE = (240, 240, 240)
    DARK_BG = (15, 23, 42)     # Slate 900

    # 1. Bords de visée tactiques (Coins type viseur militaire)
    pad = 16
    corner_len = 24
    thick = 2
    for (x, y, dx, dy) in [
        (pad, pad, 1, 1),
        (w - pad, pad, -1, 1),
        (pad, h - pad, 1, -1),
        (w - pad, h - pad, -1, -1)
    ]:
        cv2.line(frame_bgr, (x, y), (x + dx * corner_len, y), CYAN, thick)
        cv2.line(frame_bgr, (x, y), (x, y + dy * corner_len), CYAN, thick)

    # 2. Bandeau supérieur : Télémétrie C2 & Cap
    header_h = 34
    cv2.rectangle(frame_bgr, (pad, pad), (w - pad, pad + header_h), DARK_BG, -1)
    cv2.rectangle(frame_bgr, (pad, pad), (w - pad, pad + header_h), CYAN, 1)

    cv2.putText(frame_bgr, "TALOS // DGA-AID C2", (pad + 10, pad + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, CYAN, 1)
    cv2.putText(frame_bgr, f"CAP: {yaw_deg:03d} DEG", (cx - 45, pad + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
    cv2.putText(frame_bgr, f"t={sim_time:4.1f}s | 500Hz", (w - pad - 120, pad + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, GREEN, 1)

    # 3. Horizon Artificiel (Gyromètre IMU asservi en temps réel)
    # Déplacement vertical selon pitch, rotation selon roll
    pitch_offset = int(np.clip(pitch_deg * 2.5, -50, 50))
    angle_rad = np.radians(roll_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    hw = 35
    x1 = int(cx - hw * cos_a)
    y1 = int(cy + pitch_offset - hw * sin_a)
    x2 = int(cx - 10 * cos_a)
    y2 = int(cy + pitch_offset - 10 * sin_a)
    cv2.line(frame_bgr, (x1, y1), (x2, y2), CYAN, 2)

    x3 = int(cx + 10 * cos_a)
    y3 = int(cy + pitch_offset + 10 * sin_a)
    x4 = int(cx + hw * cos_a)
    y4 = int(cy + pitch_offset + hw * sin_a)
    cv2.line(frame_bgr, (x3, y3), (x4, y4), CYAN, 2)
    cv2.circle(frame_bgr, (cx, cy), 3, CYAN, -1)

    # 4. Réticule central tactique
    reticle_color = GREEN if dist_to_goal > 0.30 else (0, 255, 255)
    cv2.circle(frame_bgr, (cx, cy), 16, reticle_color, 1)
    cv2.line(frame_bgr, (cx - 24, cy), (cx - 18, cy), reticle_color, 1)
    cv2.line(frame_bgr, (cx + 18, cy), (cx + 24, cy), reticle_color, 1)
    cv2.line(frame_bgr, (cx, cy - 24), (cx, cy - 18), reticle_color, 1)
    cv2.line(frame_bgr, (cx, cy + 18), (cx, cy + 24), reticle_color, 1)

    # 5. Bandeau inférieur gauche : Diagnostics Défense
    info_w, info_h = 160, 68
    iy = h - pad - info_h
    cv2.rectangle(frame_bgr, (pad, iy), (pad + info_w, h - pad), DARK_BG, -1)
    cv2.rectangle(frame_bgr, (pad, iy), (pad + info_w, h - pad), CYAN, 1)

    cv2.putText(frame_bgr, f"CIBLE : {dist_to_goal:4.2f}m", (pad + 8, iy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, WHITE, 1)
    cv2.putText(frame_bgr, f"VITESSE: {speed_ms:4.2f} m/s", (pad + 8, iy + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, CYAN, 1)
    cv2.putText(frame_bgr, f"PITCH/ROLL: {pitch_deg:+.1f} / {roll_deg:+.1f}", (pad + 8, iy + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.35, GREEN, 1)
    cv2.putText(frame_bgr, f"REFLEXE ANTI-CHUTE: ON", (pad + 8, iy + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.32, GREEN, 1)

    # 6. Mini-radar LiDAR 3D dans le coin inférieur droit
    rw, rh = 135, 115
    rx = w - pad - rw
    ry = h - pad - rh
    cv2.rectangle(frame_bgr, (rx, ry), (rx + rw, ry + rh), DARK_BG, -1)
    cv2.rectangle(frame_bgr, (rx, ry), (rx + rw, ry + rh), CYAN, 1)

    rcx, rcy = rx + rw // 2, ry + rh - 12
    cv2.circle(frame_bgr, (rcx, rcy), 4, CYAN, -1)
    scale = 22.0
    for r_m in [1.0, 2.0, 3.0]:
        cv2.circle(frame_bgr, (rcx, rcy), int(r_m * scale), (51, 65, 85), 1)

    angles = np.linspace(-np.radians(80), np.radians(80), len(ranges))
    for a, dist in zip(angles, ranges):
        d_c = min(dist, 3.2)
        px = int(rcx - d_c * np.sin(a) * scale)
        py = int(rcy - d_c * np.cos(a) * scale)
        color = (0, 0, 255) if dist < 1.8 else GREEN
        cv2.line(frame_bgr, (rcx, rcy), (px, py), color, 1)

    ax = int(rcx - 2.4 * np.sin(best_angle) * scale)
    ay = int(rcy - 2.4 * np.cos(best_angle) * scale)
    cv2.arrowedLine(frame_bgr, (rcx, rcy), (ax, ay), (0, 255, 255), 2, tipLength=0.3)
    cv2.putText(frame_bgr, "RADAR LIVOX 3D", (rx + 8, ry + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, CYAN, 1)

if __name__ == "__main__":
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    test_img[:] = (25, 30, 35)
    ranges = np.ones(45) * 2.5
    ranges[20:25] = 1.2
    draw_pro_tactical_hud(test_img, ranges, 0.15, 35, -4.2, 2.1, 2.85, 0.52, 6.4)
    cv2.imwrite("preview_pro_tactical_hud.png", test_img)
    print("Test pro tactical HUD generated: preview_pro_tactical_hud.png")
