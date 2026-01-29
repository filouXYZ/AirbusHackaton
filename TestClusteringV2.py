import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import lidar_utils  # On utilise le fichier fourni par Airbus

# --- CONFIGURATION ---
# Mapping des couleurs fourni dans la consigne (R, G, B)
COLORS = {
    0: (38, 23, 180),  # Antenna
    1: (177, 132, 47),  # Cable
    2: (129, 81, 97),  # Electric Pole
    3: (66, 132, 9)  # Wind Turbine
}

CLASS_NAMES = {0: "Antenna", 1: "Cable", 2: "Pole", 3: "Turbine"}

# Paramètres de Clustering (A ajuster !)
# eps: distance max entre deux points pour dire qu'ils sont liés
# min_samples: nombre min de points pour faire un objet
CLUSTER_PARAMS = {
    0: {'eps': 0.5, 'min_samples': 5},  # Antenna (objets denses)
    1: {'eps': 1.5, 'min_samples': 10},  # Cable (très étendu, points espacés)
    2: {'eps': 0.5, 'min_samples': 10},  # Pole
    3: {'eps': 2.0, 'min_samples': 20}  # Turbine (très grand)
}


def get_bbox_from_points(points):
    """
    Calcule une boîte englobante simple (alignée sur les axes X/Y/Z)
    pour un nuage de points donné.
    """
    min_coords = points.min(axis=0)  # [min_x, min_y, min_z]
    max_coords = points.max(axis=0)  # [max_x, max_y, max_z]

    # Centre = moyenne entre min et max
    center = (min_coords + max_coords) / 2

    # Dimensions = max - min
    dims = max_coords - min_coords

    return center, dims


def process_file(file_path):
    print(f"Traitement du fichier : {file_path}")

    # 1. Charger les données
    df = lidar_utils.load_h5_data(file_path)

    # 2. Récupérer la liste des frames (poses)
    poses = lidar_utils.get_unique_poses(df)
    print(f"--> Nombre de frames trouvées : {len(poses)}")

    # Pour le test, on ne fait que la première frame (change 0 par i dans une boucle plus tard)
    frame_idx = 0
    pose = poses.iloc[frame_idx]

    # 3. Filtrer les points de cette frame spécifique
    frame_df = lidar_utils.filter_by_pose(df, pose)

    # 4. Convertir en Cartésien (X, Y, Z)
    # lidar_utils renvoie une matrice numpy Nx3
    xyz = lidar_utils.spherical_to_local_cartesian(frame_df)

    # On ajoute les infos de couleur/classe au tableau numpy pour faciliter le tri
    # On récupère R, G, B du dataframe
    rgb = frame_df[['r', 'g', 'b']].values

    detected_objects = []

    # 5. Boucle sur chaque classe (Antenne, Cable, etc.)
    for class_id, color in COLORS.items():
        # Trouver les points qui correspondent à cette couleur EXACTE
        mask = (rgb[:, 0] == color[0]) & (rgb[:, 1] == color[1]) & (rgb[:, 2] == color[2])
        class_points = xyz[mask]

        if len(class_points) == 0:
            continue

        print(f"   Classe {CLASS_NAMES[class_id]} : {len(class_points)} points trouvés.")

        # 6. CLUSTERING (DBSCAN)
        # C'est ici que la magie opère : on sépare les objets distincts
        params = CLUSTER_PARAMS[class_id]
        clustering = DBSCAN(eps=params['eps'], min_samples=params['min_samples']).fit(class_points)

        labels = clustering.labels_
        unique_labels = set(labels)

        # Pour chaque cluster trouvé (sauf le bruit -1)
        for label in unique_labels:
            if label == -1: continue  # C'est du bruit, on ignore

            # Extraire les points de CET objet précis
            object_points = class_points[labels == label]

            # 7. Calculer la Boîte (Ground Truth)
            center, dims = get_bbox_from_points(object_points)

            # Stocker le résultat
            detected_objects.append({
                'frame_idx': frame_idx,
                'class_id': class_id,
                'class_name': CLASS_NAMES[class_id],
                'center_x': center[0], 'center_y': center[1], 'center_z': center[2],
                'size_x': dims[0], 'size_y': dims[1], 'size_z': dims[2],
                'num_points': len(object_points)
            })

    # 8. Afficher les résultats
    results_df = pd.DataFrame(detected_objects)
    print("\n--- Objets Détectés (Vérité Terrain Reconstruite) ---")
    print(results_df)

    return results_df


if __name__ == "__main__":
    # Remplace par le chemin vers ton fichier .h5
    # Exemple : "train/Hackathon_2026_Scene01.h5"
    process_file("trainingdata/scene_1.h5")