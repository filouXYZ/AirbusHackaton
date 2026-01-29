import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import lidar_utils  # Le fichier fourni par Airbus
import os

# --- 1. CONFIGURATION DES COULEURS (Données Consigne) ---
COLORS = {
    0: (38, 23, 180),  # Antenna (Bleu foncé)
    1: (177, 132, 47),  # Cable (Doré)
    2: (129, 81, 97),  # Electric Pole (Gris/Rose)
    3: (66, 132, 9)  # Wind Turbine (Vert)
}
CLASS_NAMES = {0: "Antenna", 1: "Cable", 2: "Pole", 3: "Turbine"}

# --- 2. PARAMÈTRES DBSCAN (À ajuster si besoin) ---
# eps : distance max entre deux points pour qu'ils soient collés (en mètres)
# min_samples : nombre minimum de points pour faire un objet valide
CLUSTER_PARAMS = {
    # Antenne : Structure fine et haute. Les points peuvent être espacés verticalement.
    # On passe eps à 2.0m ou 3.0m pour recoller les morceaux.
    0: {'eps': 3.0, 'min_samples': 5},
    # Câble : Le pire cas. C'est une ligne très fine, les points sont très rares.
    # Il faut un eps énorme pour relier les points le long du fil.
    1: {'eps': 5.0, 'min_samples': 3},
    # Poteau : Souvent plus dense que les câbles, mais 0.8 était trop juste.
    #2: {'eps': 1, 'min_samples': 5}
    #2: {'eps': 1.3, 'min_samples': 10},
    2: {'eps': 1.2, 'min_samples': 12},
    # Turbine : C'est immense. On peut être large.
    3: {'eps': 4.5, 'min_samples': 17}
    #3: {'eps': 4.0, 'min_samples': 10}
}


# --- FONCTIONS UTILITAIRES ---

def clean_lidar_points(df, max_distance_m=150.0):
    """Nettoie les points invalides (distance=0) et trop lointains."""
    df = df[df["distance_cm"] > 0]  # Retire les erreurs de tir
    return df.reset_index(drop=True)


def get_bbox_from_points(points):
    """Calcule le centre et la taille d'une boîte autour des points."""
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    center = (min_coords + max_coords) / 2
    dims = max_coords - min_coords  # Largeur, Longueur, Hauteur
    return center, dims


def process_file(file_path):
    print(f"Traitement du fichier : {file_path}")

    # 1. Charger le fichier
    try:
        df_full = lidar_utils.load_h5_data(file_path)
    except Exception as e:
        print(f"Erreur de chargement : {e}")
        return

    # 2. Nettoyage GLOBAL (Très important !)
    df_full = clean_lidar_points(df_full)

    # 3. Récupérer la liste des frames (poses)
    poses = lidar_utils.get_unique_poses(df_full)
    print(f"--> {len(poses)} frames trouvées.")

    detected_objects = []

    # 4. BOUCLE SUR CHAQUE FRAME
    # (On utilise iterrows pour ne traiter qu'une frame à la fois et sauver la RAM)
    for i, pose_row in poses.iterrows():

        # Filtrer pour n'avoir que les points de CETTE frame
        frame_df = lidar_utils.filter_by_pose(df_full, pose_row)

        if len(frame_df) == 0: continue

        # Convertir en X, Y, Z (Mètres)
        xyz = lidar_utils.spherical_to_local_cartesian(frame_df)

        # Récupérer les couleurs pour le tri
        rgb = frame_df[['r', 'g', 'b']].values

        # --- ANALYSE PAR CLASSE ---
        for class_id, color in COLORS.items():
            # Masque : On ne garde que les points de la bonne couleur
            mask = (rgb[:, 0] == color[0]) & (rgb[:, 1] == color[1]) & (rgb[:, 2] == color[2])
            class_points = xyz[mask]

            if len(class_points) == 0:
                continue  # Aucun objet de cette classe dans cette frame

            # --- LE CŒUR DU RÉACTEUR : DBSCAN ---
            params = CLUSTER_PARAMS[class_id]
            clustering = DBSCAN(eps=params['eps'], min_samples=params['min_samples']).fit(class_points)

            labels = clustering.labels_
            unique_labels = set(labels)

            for label in unique_labels:
                if label == -1: continue  # -1 = Bruit (points isolés), on jette

                # Extraire l'objet unique
                object_points = class_points[labels == label]

                # Calculer sa boîte
                center, dims = get_bbox_from_points(object_points)

                # Sauvegarder l'info
                detected_objects.append({
                    # --- Identifiants Frame (Requis) ---
                    'pose_index': i,  # Utile pour toi (pour retrouver l'image)
                    'ego_x': pose_row['ego_x'],  # Requis par les juges
                    'ego_y': pose_row['ego_y'],  # Requis par les juges
                    'ego_z': pose_row['ego_z'],  # Requis par les juges
                    'ego_yaw': pose_row['ego_yaw'],  # Requis par les juges
                    # --- Géométrie de la Boîte (Noms imposés par les juges) ---
                    'bbox_center_x': center[0],
                    'bbox_center_y': center[1],
                    'bbox_center_z': center[2],

                    'bbox_width': dims[0],  # size_x
                    'bbox_length': dims[1],  # size_y
                    'bbox_height': dims[2],  # size_z
                    'bbox_yaw': 0.0,  # On ne gère pas la rotation pour l'instant (Axis Aligned)

                    # --- Classe (Requis) ---
                    'class_id': class_id,
                    'class_label': CLASS_NAMES[class_id],

                    # --- Nombre de points de l'objet (Utile pour le tri, à supprimer plus tard) ---
                    'num_points': len(object_points)
                })

        # Petit print pour voir que ça avance
        if i % 10 == 0: print(f"   Frame {i}/{len(poses)} traitée...")

    # 5. Sauvegarde finale en CSV
    if detected_objects:
        results_df = pd.DataFrame(detected_objects)
        output_name = "verite_terrain_generee.csv"
        results_df.to_csv(output_name, index=False)
        print(f"\n✅ Terminé ! J'ai trouvé {len(results_df)} objets.")
        print(f"📁 Résultats sauvegardés dans : {output_name}")
        print(results_df.head())  # Affiche les 5 premiers
    else:
        print("\n❌ Aucun objet détecté (vérifie tes couleurs ou tes seuils DBSCAN).")


if __name__ == "__main__":
    # CHANGE CE CHEMIN vers ton vrai fichier .h5
    mon_fichier = "trainingdata/scene_1.h5"

    if os.path.exists(mon_fichier):
        process_file(mon_fichier)
    else:
        print(f"Fichier introuvable : {mon_fichier}")