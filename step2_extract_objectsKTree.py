import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm
from scipy.spatial import cKDTree
import lidar_utils

OUTPUT_DIR = "dataset_prepared"
CLASS_NAMES = {0: "Antenna", 1: "Cable", 2: "Pole", 3: "Turbine"}


def extract_objects(csv_path, data_root_dir):
    print(f"📂 Lecture du CSV : {csv_path}")
    try:
        df_labels = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Erreur lecture CSV : {e}")
        return

    if 'file_source' not in df_labels.columns:
        print("❌ ERREUR : Colonne 'file_source' absente.")
        return

    # Création des dossiers
    for class_name in CLASS_NAMES.values():
        os.makedirs(os.path.join(OUTPUT_DIR, class_name), exist_ok=True)

    grouped = df_labels.groupby('file_source')
    print(f"🚀 Début de l'extraction sur {len(grouped)} fichiers...")

    total_extracted = 0

    # BARRE 1 : Progression globale (Fichier par Fichier)
    # position=0 permet de la garder en haut
    outer_bar = tqdm(grouped, desc="Global", position=0)

    for filename, group in outer_bar:
        file_path = os.path.join(data_root_dir, filename)

        if not os.path.exists(file_path):
            continue

        # Mise à jour du texte de la barre du haut
        outer_bar.set_description(f"Traitement : {filename}")

        # 1. Chargement & KD-Tree (Partie lourde)
        try:
            df_lidar = lidar_utils.load_h5_data(file_path)
            df_lidar = df_lidar[df_lidar["distance_cm"] > 0]
            xyz = lidar_utils.spherical_to_local_cartesian(df_lidar)
            tree = cKDTree(xyz)  # Construction de l'arbre
        except Exception as e:
            print(f"❌ Erreur H5 {filename}: {e}")
            continue

        # BARRE 2 : Progression Interne (Objet par Objet dans ce fichier)
        # leave=False signifie que cette barre disparaîtra quand le fichier est fini
        inner_bar = tqdm(group.iterrows(), total=len(group), desc="Extraction Objets", leave=False, position=1)

        for idx, row in inner_bar:
            cx, cy, cz = row['bbox_center_x'], row['bbox_center_y'], row['bbox_center_z']
            dx, dy, dz = row['bbox_width'], row['bbox_length'], row['bbox_height']
            label = row['class_label']

            # Recherche KD-Tree
            radius = np.sqrt((dx / 2) ** 2 + (dy / 2) ** 2 + (dz / 2) ** 2)
            candidate_indices = tree.query_ball_point([cx, cy, cz], r=radius * 1.1)

            if len(candidate_indices) == 0:
                continue

            local_points = xyz[candidate_indices]

            # Masque rectangulaire précis
            min_x, max_x = cx - dx / 2, cx + dx / 2
            min_y, max_y = cy - dy / 2, cy + dy / 2
            min_z, max_z = cz - dz / 2, cz + dz / 2

            mask = (
                    (local_points[:, 0] >= min_x) & (local_points[:, 0] <= max_x) &
                    (local_points[:, 1] >= min_y) & (local_points[:, 1] <= max_y) &
                    (local_points[:, 2] >= min_z) & (local_points[:, 2] <= max_z)
            )

            points_inside = local_points[mask]

            if len(points_inside) < 5:
                continue

            # Normalisation & Sauvegarde
            points_inside[:, 0] -= cx
            points_inside[:, 1] -= cy
            points_inside[:, 2] -= cz

            save_name = f"{label}_{idx}.npy"
            np.save(os.path.join(OUTPUT_DIR, label, save_name), points_inside)
            total_extracted += 1

    print(f"\n✅ TERMINÉ ! {total_extracted} objets extraits.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()

    extract_objects(args.csv, args.data_dir)