"""Signature thermique du robot : modele a une capacite thermique (toutes les valeurs sont des ASSUMPTION).

  C dT/dt = f_perte * P_elec - hA (T - T_air)

  - P_elec : puissance electrique (modele energy.py, ou mesure MuJoCo) ;
  - f_perte = 0,85 : part de la puissance dissipee en chaleur DANS le robot (le reste = travail fourni au sol / a l'air) ;
  - C = 9 000 J/K : ~10 kg d'aluminium + batterie qui chauffent (sur ~35 kg) ;
  - hA = 12 W/K : convection naturelle h ~ 10 W/m2/K sur ~1,2 m2 de surface ;
  -> constante de temps C/hA ~ 750 s, ecart d'equilibre ~ 14 K a 200 W.

Portee de detection par une camera thermique (LWIR non refroidie) : on suppose R = R_REF * sqrt(dT / DT_REF),
R_REF = 1 500 m pour dT = 10 K. C'est un ORDRE DE GRANDEUR (cible de taille humaine, ciel degage) ; en foret le
couvert limite la ligne de vue bien avant. Seule la comparaison entre profils de mission est significative.
"""
import numpy as np

C_J_PER_K = 9000.0
HA_W_PER_K = 12.0
LOSS_FRACTION = 0.85
R_REF_M, DT_REF_K = 1500.0, 10.0


class ThermalModel:
    def __init__(self, t_air=10.0):
        self.t_air = t_air
        self.dT = 0.0                      # ecart a l'air (K) ; robot a l'equilibre au depart (sortie d'abri)

    def step(self, p_elec_w, dt):
        self.dT += dt * (LOSS_FRACTION * p_elec_w - HA_W_PER_K * self.dT) / C_J_PER_K
        return self.dT

    @staticmethod
    def detection_range_m(dT):
        return float(R_REF_M * np.sqrt(max(dT, 0.0) / DT_REF_K))

    @staticmethod
    def cooldown_time_s(dT_from, dT_to):
        """Temps a l'arret (P = base) pour descendre de dT_from a dT_to."""
        from .energy import P_BASE_W
        eq = LOSS_FRACTION * P_BASE_W / HA_W_PER_K
        if dT_to <= eq:
            return float("inf")
        tau = C_J_PER_K / HA_W_PER_K
        return float(tau * np.log((dT_from - eq) / (dT_to - eq))) if dT_from > dT_to else 0.0
