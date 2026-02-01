import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import lidar_utils
import os
import argparse
import gc
import glob
from tqdm import tqdm  # Pour la barre de progression

# --- 1. CONFIGURATION FINALE ---
COLORS = {
    0: (38, 23, 180),  # Antenna
    1: (177, 132, 47),  # Cable
    2: (129, 81, 97),  # Pole
    3: (66, 132, 9)  # Turbine
}
CLASS_NAMES = {0: "Antenna", 1: "Cable", 2: "Pole", 3: "Turbine"}

# Tes paramètres validés
CLUSTER_PARAMS = {
    0: {'eps': 3.0, 'min_samples': 5},  # Antenna
    1: {'eps': 5.0, 'min_samples': 3},  # Cable
    2: {'eps': 1.2, 'min_samples': 12},  # Pole (Ton choix final)
    3: {'eps': 4.5, 'min_samples': 17}  # Turbine (Correction anti-aspiration)
}


# --- 2. FONCTIONS ---

def clean_lidar_points(df):
    df = df[df["distance_cm"] > 0]
    return df.reset_index(drop=True)


def get_bbox_from_points(points):
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    center = (min_coords + max_coords) / 2
    dims = max_coords - min_coords
    return center, dims


def process_dataset(input_dir, output_file):
    # Récupérer tous les fichiers .h5 du dossier
    files = glob.glob(os.path.join(input_dir, "*.h5"))
    print(f"🚀 Démarrage du traitement de masse : {len(files)} fichiers trouvés.")

    all_detected_objects = []

    # Boucle sur chaque fichier avec une barre de progression
    for file_path in tqdm(files, desc="Traitement des fichiers"):
        try:
            df_full = lidar_utils.load_h5_data(file_path)
        except Exception as e:
            print(f"❌ Erreur lecture {file_path}: {e}")
            continue

        df_full = clean_lidar_points(df_full)
        poses = lidar_utils.get_unique_poses(df_full)

        # Boucle sur les frames du fichier
        for i, pose_row in poses.iterrows():
            frame_df = lidar_utils.filter_by_pose(df_full, pose_row)
            if len(frame_df) == 0: continue

            xyz = lidar_utils.spherical_to_local_cartesian(frame_df)
            rgb = frame_df[['r', 'g', 'b']].values

            for class_id, color in COLORS.items():
                mask = (rgb[:, 0] == color[0]) & (rgb[:, 1] == color[1]) & (rgb[:, 2] == color[2])
                class_points = xyz[mask]

                if len(class_points) == 0: continue

                # SÉCURITÉ ANTI-CRASH RAM
                MAX_POINTS = 70000
                if len(class_points) > MAX_POINTS:
                    print(f"⚠️ Trop de points ({len(class_points)}), on échantillonne...")
                    # On prend des indices au hasard
                    indices = np.random.choice(len(class_points), MAX_POINTS, replace=False)
                    class_points = class_points[indices]

                params = CLUSTER_PARAMS[class_id]
                clustering = DBSCAN(eps=params['eps'], min_samples=params['min_samples']).fit(class_points)

                labels = clustering.labels_
                unique_labels = set(labels)

                for label in unique_labels:
                    if label == -1: continue

                    object_points = class_points[labels == label]
                    center, dims = get_bbox_from_points(object_points)

                    # --- FILTRES pour empecher les objets d'etre inférieur à X metres ---
                    if class_id == 0 and dims[2] < 1.0: continue  # Antenne naine
                    if class_id == 1 and dims[0] < 0.5 and dims[1] < 0.5: continue  # Câble poussière
                    if class_id == 2 and dims[2] < 2.0: continue  # Poteau nain
                    if class_id == 3 and dims[2] < 3.0: continue  # Turbine naine
                    # ---------------------------

                    # Ajout à la liste globale
                    all_detected_objects.append({
                        'file_source': os.path.basename(file_path),  # Important pour le step2
                        'pose_index': i,
                        'ego_x': pose_row['ego_x'],
                        'ego_y': pose_row['ego_y'],
                        'ego_z': pose_row['ego_z'],
                        'ego_yaw': pose_row['ego_yaw'],

                        # Format Livrable
                        'bbox_center_x': center[0],
                        'bbox_center_y': center[1],
                        'bbox_center_z': center[2],

                        'bbox_width': dims[0],
                        'bbox_length': dims[1],
                        'bbox_height': dims[2],
                        'bbox_yaw': 0.0,

                        'class_id': class_id,
                        'class_label': CLASS_NAMES[class_id],

                        # --- Nombre de points de l'objet (Utile pour le tri, à supprimer plus tard) ---
                        'num_points': len(object_points)
                    })
            # Petit print pour voir que ça avance
            if i % 10 == 0: print(f"   Frame {i}/{len(poses)} traitée...")
            del frame_df

        # Nettoyage mémoire entre chaque fichier
        del df_full
        gc.collect()

    # Sauvegarde finale
    if all_detected_objects:
        results_df = pd.DataFrame(all_detected_objects)
        results_df.to_csv(output_file, index=False)
        print(f"\n✅ TERMINÉ ! Grand CSV généré : {output_file}")
        print(f"📊 Total objets trouvés : {len(results_df)}")
    else:
        print("\n⚠️ Aucun objet trouvé dans tout le dossier.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Dossier contenant TOUS les .h5")
    parser.add_argument("--output", default="Train_Labels.csv", help="Nom du fichier CSV final")
    args = parser.parse_args()

    if os.path.exists(args.input_dir):
        process_dataset(args.input_dir, args.output)
    else:
        print("❌ Le dossier d'entrée n'existe pas.")