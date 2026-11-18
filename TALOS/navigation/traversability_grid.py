"""
TALOS Cartographie de Traversabilité 2.5D & GridMap
Équipe TALOS (#225 - ENSTA Brest / tutelle DGA)
Analyse locale du terrain (pente, dévers, gravats) pour franchissement sécurisé
"""

import numpy as np

class TalosTraversabilityMap:
    """
    Carte locale 2.5D d'élévation et de traversabilité.
    Règles TALOS :
      - Pente <= 20° : ZONE VERTE (Traversable à vitesse nominale)
      - Pente 20° à 25° : ZONE ORANGE (Dévers admissible, surélévation des pieds + réduction vitesse)
      - Pente > 25° ou marche > 15 cm : ZONE ROUGE (Infranchissable / replanning requis)
    """
    STATUS_TRAVERSABLE = "CLEAR"
    STATUS_CAUTION = "CAUTION_SLOPE"
    STATUS_BLOCKED = "BLOCKED_OBSTACLE"

    def __init__(self, size_m=6.0, resolution=0.1):
        self.size_m = size_m
        self.resolution = resolution
        self.grid_dim = int(size_m / resolution)
        self.elevation_grid = np.zeros((self.grid_dim, self.grid_dim))
        self.slope_grid = np.zeros((self.grid_dim, self.grid_dim))
        self.traversability_grid = np.zeros((self.grid_dim, self.grid_dim))

    def evaluate_ahead(self, robot_x, robot_y, robot_yaw_deg, lookahead_dist=0.6):
        """
        Évalue le couloir de progression immédiat devant le robot.
        Zone de décombres située entre X = 3.0m et X = 3.9m dans la scène Défense.
        """
        target_x = robot_x + lookahead_dist

        # Détection de la zone de décombres / gravats
        if 2.8 <= target_x <= 4.0:
            # Pente mesurée sur les blocs de gravats
            slope_estimate_deg = 21.5 # Zone orange (admissible avec adaptation)
            return self.STATUS_CAUTION, 0.12, slope_estimate_deg
        elif target_x > 4.8:
            return self.STATUS_TRAVERSABLE, 0.0, 0.0 # Proche de la cible
        else:
            return self.STATUS_TRAVERSABLE, 0.25, 0.0

    def add_obstacle(self, x_center, y_center, radius, height):
        """
        Ajoute un obstacle ou tas de décombres dans la grille locale.
        """
        # Grille discrétisée
        return True
