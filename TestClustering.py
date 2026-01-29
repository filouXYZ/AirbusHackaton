import numpy as np
from sklearn.cluster import DBSCAN
import lidar_utils # Le fichier fourni par Airbus

# 1. Charger un fichier
df = lidar_utils.load_h5_data("trainingdata/scene_1.h5")

# 2. Filtrer juste les points qui sont des "Poteaux" (Class ID 2)
# Les couleurs pour poteau sont R=129, G=81, B=97
poteaux_points = df[
    (df['r'] == 129) &
    (df['g'] == 81) &
    (df['b'] == 97)
]

# 3. Convertir en X, Y, Z (Tu devras utiliser les fonctions de lidar_utils)
# Supposons que tu aies maintenant une variable 'xyz_points'

# 4. Clustering (La magie)
# eps=0.5 signifie "cherche des points à 50cm les uns des autres"
# min_samples=10 signifie "il faut au moins 10 points pour faire un objet"
clustering = DBSCAN(eps=0.5, min_samples=10).fit(xyz_points)

# 5. Regarder combien d'objets on a trouvés
labels = clustering.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"J'ai trouvé {n_clusters} poteaux dans cette frame !")