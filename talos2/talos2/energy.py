"""Bilan energetique. MODELE a hypotheses affichees (params.py), calibre sur la donnee constructeur
'~2 h d'autonomie' puis a REMPLACER par l'integrale couple x vitesse mesuree sous MuJoCo."""
import numpy as np

G = 9.81
ETA = 0.25            # ASSUMPTION rendement chaine de traction pour le travail en montee
P_BASE_W = 70.0       # ASSUMPTION calcul + capteurs + maintien debout
COT_FLAT = 0.41       # CALIBRE: donne ~2 h a 1 m/s a vide sur terrain plat (SPEC constructeur: jusqu'a 2 h)
MUD_FACTOR = 1.5      # ASSUMPTION surcout energetique dans la boue


def power_w(p, v, slope, in_mud, payload_kg=None):
    """Puissance electrique (W) a la vitesse v (m/s), pente (m/m), boue oui/non."""
    m = p.mass_kg + (p.payload_kg if payload_kg is None else payload_kg)
    walk = COT_FLAT * m * G * abs(v)
    climb = max(0.0, slope) * m * G * abs(v) / ETA        # on ne recupere rien en descente
    return P_BASE_W + (walk + climb) * (MUD_FACTOR if in_mud else 1.0)


def calibration_autonomy_hours(p, v=1.0):
    """Autonomie sur terrain plat a la vitesse v. A comparer au ~2 h annonce par le constructeur."""
    return p.battery_wh / power_w(p, v, 0.0, False)
