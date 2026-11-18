#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Générateur de Logo Vectoriel Pur pour TALOS
Sans IA générative : dessin géométrique vectoriel haute précision sous Pillow
"""

import math
from PIL import Image, ImageDraw, ImageFont

# Canvas ultra haute résolution avec supersampling 2x (2048 -> 1024)
SIZE = 2048
CENTER = SIZE // 2

# Palette de couleurs "Defense & Stealth Robotics"
BG_COLOR = (10, 14, 23)        # #0A0E17 (Bleu nuit profond / furtif)
CYAN_GLOW = (0, 229, 255)      # #00E5FF (Cyan tactique électrique)
CYAN_DARK = (0, 102, 180)      # #0066B4 (Cyan intermédiaire)
WHITE = (255, 255, 255)        # Blanc pur
TITANIUM = (160, 178, 200)     # Titane brossé clair
DARK_STEEL = (24, 34, 52)      # Acier sombre pour contraste
GOLD_ACCENT = (255, 180, 0)    # Or discret pour rappel DGA/Talos mythologique

im = Image.new("RGBA", (SIZE, SIZE), BG_COLOR)
draw = ImageDraw.Draw(im)

# 1. Anneau extérieur tactique (réticule de visée / optique)
R_OUTER = 920
R_INNER = 880

# Cercle de fond subtil
draw.ellipse(
    [CENTER - R_OUTER, CENTER - R_OUTER, CENTER + R_OUTER, CENTER + R_OUTER],
    outline=(20, 32, 50),
    width=6
)

# Anneau de précision cyan
draw.ellipse(
    [CENTER - R_INNER, CENTER - R_INNER, CENTER + R_INNER, CENTER + R_INNER],
    outline=CYAN_DARK,
    width=4
)

# Graduations tactiques circulaires (ticks militaires à 360°)
for deg in range(0, 360, 10):
    rad = math.radians(deg)
    is_major = (deg % 30 == 0)
    is_cardinal = (deg % 90 == 0)

    tick_len = 45 if is_cardinal else (28 if is_major else 14)
    tick_width = 8 if is_cardinal else (5 if is_major else 3)
    tick_col = CYAN_GLOW if is_cardinal else (TITANIUM if is_major else (40, 55, 80))

    r1 = R_INNER - 8
    r2 = r1 - tick_len

    x1 = CENTER + r1 * math.cos(rad)
    y1 = CENTER + r1 * math.sin(rad)
    x2 = CENTER + r2 * math.cos(rad)
    y2 = CENTER + r2 * math.sin(rad)

    draw.line([x1, y1, x2, y2], fill=tick_col, width=tick_width)

# 4 Chevrons cardinaux (Haut, Bas, Gauche, Droite)
for deg in [0, 90, 180, 270]:
    rad = math.radians(deg)
    r_chev = R_INNER + 20
    cx = CENTER + r_chev * math.cos(rad)
    cy = CENTER + r_chev * math.sin(rad)
    # Petit triangle tactique
    dr = 25
    dw = 18
    p1 = (cx + dr * math.cos(rad), cy + dr * math.sin(rad))
    p2 = (cx - dr * math.cos(rad) + dw * math.sin(rad), cy - dr * math.sin(rad) - dw * math.cos(rad))
    p3 = (cx - dr * math.cos(rad) - dw * math.sin(rad), cy - dr * math.sin(rad) + dw * math.cos(rad))
    draw.polygon([p1, p2, p3], fill=CYAN_GLOW)


# 2. Le Bouclier Hexagonal Stealth (Inspiré de l'armure TALOS)
# Coordonnées du bouclier central facetté
SHIELD_TOP_Y = CENTER - 540
SHIELD_BOT_Y = CENTER + 420
SHIELD_W_TOP = 460
SHIELD_W_MID = 540
SHIELD_MID_Y = CENTER - 60

# Grand bouclier extérieur
poly_shield_ext = [
    (CENTER, SHIELD_TOP_Y - 40),
    (CENTER + SHIELD_W_TOP, SHIELD_TOP_Y + 80),
    (CENTER + SHIELD_W_MID, SHIELD_MID_Y),
    (CENTER + 320, SHIELD_BOT_Y - 80),
    (CENTER, SHIELD_BOT_Y + 120),
    (CENTER - 320, SHIELD_BOT_Y - 80),
    (CENTER - SHIELD_W_MID, SHIELD_MID_Y),
    (CENTER - SHIELD_W_TOP, SHIELD_TOP_Y + 80),
]

# Fond du bouclier (acier sombre)
draw.polygon(poly_shield_ext, fill=DARK_STEEL, outline=CYAN_GLOW, width=8)

# Bouclier intérieur avec effet facetté
scale = 0.88
poly_shield_int = [
    (CENTER + (x - CENTER) * scale, CENTER - 40 + (y - (CENTER - 40)) * scale)
    for x, y in poly_shield_ext
]
draw.polygon(poly_shield_int, fill=(15, 22, 35), outline=CYAN_DARK, width=5)

# Ligne de crête centrale verticale
draw.line([(CENTER, SHIELD_TOP_Y + 20), (CENTER, SHIELD_BOT_Y + 50)], fill=(35, 50, 75), width=4)


# 3. L'EMBLÈME 'T' GÉOMÉTRIQUE CENTRAL (TALOS)
# Style mecha / stealth ultra précis
T_TOP_Y = CENTER - 320
T_BAR_H = 130
T_STEM_W = 150
T_STEM_BOT_Y = CENTER + 280

# Aile gauche de la barre supérieure du T
poly_t_left = [
    (CENTER - 10, T_TOP_Y),
    (CENTER - 380, T_TOP_Y),
    (CENTER - 420, T_TOP_Y + 60),
    (CENTER - 340, T_TOP_Y + T_BAR_H),
    (CENTER - (T_STEM_W // 2), T_TOP_Y + T_BAR_H),
    (CENTER - 10, T_TOP_Y + T_BAR_H),
]
draw.polygon(poly_t_left, fill=TITANIUM, outline=WHITE, width=4)

# Aile droite de la barre supérieure du T (symétrique, accentuée cyan)
poly_t_right = [
    (CENTER + 10, T_TOP_Y),
    (CENTER + 380, T_TOP_Y),
    (CENTER + 420, T_TOP_Y + 60),
    (CENTER + 340, T_TOP_Y + T_BAR_H),
    (CENTER + (T_STEM_W // 2), T_TOP_Y + T_BAR_H),
    (CENTER + 10, T_TOP_Y + T_BAR_H),
]
draw.polygon(poly_t_right, fill=WHITE, outline=CYAN_GLOW, width=4)

# Pilier central du T (effilé vers le bas comme une lame d'armure)
poly_t_stem_left = [
    (CENTER - 10, T_TOP_Y + T_BAR_H + 10),
    (CENTER - (T_STEM_W // 2), T_TOP_Y + T_BAR_H + 10),
    (CENTER - (T_STEM_W // 2) + 20, T_STEM_BOT_Y - 80),
    (CENTER - 10, T_STEM_BOT_Y),
]
draw.polygon(poly_t_stem_left, fill=TITANIUM, outline=WHITE, width=4)

poly_t_stem_right = [
    (CENTER + 10, T_TOP_Y + T_BAR_H + 10),
    (CENTER + (T_STEM_W // 2), T_TOP_Y + T_BAR_H + 10),
    (CENTER + (T_STEM_W // 2) - 20, T_STEM_BOT_Y - 80),
    (CENTER + 10, T_STEM_BOT_Y),
]
draw.polygon(poly_t_stem_right, fill=WHITE, outline=CYAN_GLOW, width=4)

# Inserts luminescents cyan au coeur du T (Core Reactor / Visière)
core_w = 40
core_y1 = T_TOP_Y + T_BAR_H + 30
core_y2 = T_STEM_BOT_Y - 120
draw.line([(CENTER, core_y1), (CENTER, core_y2)], fill=CYAN_GLOW, width=12)

# Point focal central lumineux
draw.ellipse([CENTER - 20, core_y1 + 40, CENTER + 20, core_y1 + 80], fill=WHITE, outline=CYAN_GLOW, width=6)

# Chevrons tactiques d'armure latéraux (G1 / Go2 wings)
for side in [-1, 1]:
    chev_x = CENTER + side * 460
    chev_y = CENTER + 140
    poly_side_chev = [
        (chev_x, chev_y - 80),
        (chev_x + side * 50, chev_y),
        (chev_x, chev_y + 80),
        (chev_x - side * 30, chev_y + 50),
        (chev_x + side * 15, chev_y),
        (chev_x - side * 30, chev_y - 50),
    ]
    draw.polygon(poly_side_chev, fill=CYAN_GLOW if side == 1 else TITANIUM)


# 4. TYPOGRAPHIE VECTORIELLE PURE : "TALOS"
# On trace des lettres géométriques parfaites au compas/règle
def draw_letter_T(cx, cy, h=140, w=100, col=WHITE, th=24):
    draw.line([(cx - w//2, cy - h//2), (cx + w//2, cy - h//2)], fill=col, width=th)
    draw.line([(cx, cy - h//2), (cx, cy + h//2)], fill=col, width=th)

def draw_letter_A(cx, cy, h=140, w=110, col=WHITE, th=24):
    p1 = (cx, cy - h//2)
    p2 = (cx - w//2, cy + h//2)
    p3 = (cx + w//2, cy + h//2)
    draw.line([p1, p2], fill=col, width=th)
    draw.line([p1, p3], fill=col, width=th)
    draw.line([(cx - w//4, cy + h//6), (cx + w//4, cy + h//6)], fill=col, width=th - 4)

def draw_letter_L(cx, cy, h=140, w=90, col=WHITE, th=24):
    draw.line([(cx - w//2, cy - h//2), (cx - w//2, cy + h//2)], fill=col, width=th)
    draw.line([(cx - w//2, cy + h//2), (cx + w//2, cy + h//2)], fill=col, width=th)

def draw_letter_O(cx, cy, h=140, w=110, col=WHITE, th=24):
    # Hexagone stylisé pour le O
    pts = [
        (cx - w//4, cy - h//2),
        (cx + w//4, cy - h//2),
        (cx + w//2, cy),
        (cx + w//4, cy + h//2),
        (cx - w//4, cy + h//2),
        (cx - w//2, cy),
    ]
    draw.polygon(pts, outline=col, width=th)

def draw_letter_S(cx, cy, h=140, w=100, col=WHITE, th=24):
    # S biseauté géométrique
    p_s = [
        (cx + w//2, cy - h//2 + 20),
        (cx + w//4, cy - h//2),
        (cx - w//4, cy - h//2),
        (cx - w//2, cy - h//4),
        (cx + w//2, cy + h//8),
        (cx + w//2, cy + h//3),
        (cx + w//4, cy + h//2),
        (cx - w//4, cy + h//2),
        (cx - w//2, cy + h//2 - 20)
    ]
    for i in range(len(p_s) - 1):
        draw.line([p_s[i], p_s[i+1]], fill=col, width=th)

# Affichage du mot "TALOS" en bas
text_y = CENTER + 650
spacing = 150
letters = [
    (draw_letter_T, CENTER - 2 * spacing),
    (draw_letter_A, CENTER - 1 * spacing),
    (draw_letter_L, CENTER),
    (draw_letter_O, CENTER + 1 * spacing),
    (draw_letter_S, CENTER + 2 * spacing),
]

for func, lx in letters:
    func(lx, text_y, h=130, w=95, col=WHITE, th=22)

# Ligne d'accent tactique sous TALOS
bar_w = 420
draw.line([(CENTER - bar_w, text_y + 90), (CENTER + bar_w, text_y + 90)], fill=CYAN_GLOW, width=6)
# Points cardinaux de la barre
draw.ellipse([CENTER - bar_w - 8, text_y + 82, CENTER - bar_w + 8, text_y + 98], fill=WHITE)
draw.ellipse([CENTER + bar_w - 8, text_y + 82, CENTER + bar_w + 8, text_y + 98], fill=WHITE)


# 5. Redimensionnement haute qualité avec antialiasing (Lanczos 2048 -> 1024)
output_path = r"C:\Users\fredb\Documents\Hackathon\talos_vector_logo.png"
final_im = im.resize((1024, 1024), Image.Resampling.LANCZOS)
final_im.save(output_path, "PNG", quality=100)
print(f"[+] Logo vectoriel genere avec succes : {output_path}")
