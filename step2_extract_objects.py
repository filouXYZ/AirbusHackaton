import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm
import lidar_utils  # Ton fichier utilitaire habituel

# Configuration des dossiers
OUTPUT_DIR = "dataset_prepared"
CLASS_NAMES = {0: "Antenna", 1: "Cable", 2: "Pole", 3: "Turbine"}


def extract_objects(csv_path, data_root_dir):
    # 1. Charger le CSV généré par Step 1
    print(f"📂 Lecture du CSV : {csv_path}")
    try:
        df_labels = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Erreur lecture CSV : {e}")
        return

    # Créer l'arborescence : dataset_prepared/Antenna, dataset_prepared/Pole, etc.
    for class_name in CLASS_NAMES.values():
        os.makedirs(os.path.join(OUTPUT_DIR, class_name), exist_ok=True)

    # 2. Grouper par fichier source pour optimiser (on ouvre le fichier .h5 une seule fois)
    grouped = df_labels.groupby('file_source')  # Assure-toi que cette colonne existe dans ton CSV step1

    print(f"🚀 Début de l'extraction sur {len(grouped)} fichiers sources...")

    total_extracted = 0

    for filename, group in tqdm(grouped):
        file_path = os.path.join(data_root_dir, filename)

        if not os.path.exists(file_path):
            print(f"⚠️ Fichier introuvable : {file_path} (Ignoré)")
            continue

        # Charger le nuage de points complet
        try:
            df_lidar = lidar_utils.load_h5_data(file_path)
            # Conversion sphérique -> cartésien global
            xyz = lidar_utils.spherical_to_local_cartesian(df_lidar)
            # On récupère aussi l'intensité ou la réflectivité si dispo, sinon juste XYZ
            # Pour l'instant, on garde XYZ. L'IA apprendra la géométrie.
        except Exception as e:
            print(f"❌ Erreur lecture H5 {filename}: {e}")
            continue

        # Pour chaque objet détecté dans ce fichier
        for idx, row in group.iterrows():
            # Récupérer la boîte englobante
            cx, cy, cz = row['bbox_center_x'], row['bbox_center_y'], row['bbox_center_z']
            dx, dy, dz = row['bbox_width'], row['bbox_length'], row['bbox_height']
            label = row['class_label']
            class_id = row['class_id']

            # Définir les limites de la boîte (Min/Max)
            # Note: On suppose ici que la rotation yaw est 0 (Axis Aligned) comme dans Step 1
            min_x, max_x = cx - dx / 2, cx + dx / 2
            min_y, max_y = cy - dy / 2, cy + dy / 2
            min_z, max_z = cz - dz / 2, cz + dz / 2

            # Masque : Garder uniquement les points DANS la boîte
            mask = (
                    (xyz[:, 0] >= min_x) & (xyz[:, 0] <= max_x) &
                    (xyz[:, 1] >= min_y) & (xyz[:, 1] <= max_y) &
                    (xyz[:, 2] >= min_z) & (xyz[:, 2] <= max_z)
            )

            points_inside = xyz[mask]

            # Vérification de sécurité (si la boîte est vide malgré le CSV)
            if len(points_inside) < 5:
                continue

            # --- NORMALISATION (CRUCIAL) ---
            # On ramène le centre de l'objet à (0,0,0)
            # L'IA doit apprendre la FORME, pas la POSITION absolue dans la forêt.
            points_inside[:, 0] -= cx
            points_inside[:, 1] -= cy
            points_inside[:, 2] -= cz

            # Sauvegarde en .npy (Format binaire très rapide pour Python)
            # Nom du fichier : class_Label_IDunique.npy
            save_name = f"{label}_{total_extracted}.npy"
            save_path = os.path.join(OUTPUT_DIR, label, save_name)

            np.save(save_path, points_inside)
            total_extracted += 1

    print(f"✅ Terminé ! {total_extracted} objets extraits dans le dossier '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Le CSV généré à l'étape 1
    parser.add_argument("--csv", required=True, help="Chemin vers le CSV généré par Step 1")
    # Le dossier où sont tes fichiers .h5 originaux
    parser.add_argument("--data_dir", required=True, help="Dossier contenant les fichiers .h5")
    args = parser.parse_args()

    extract_objects(args.csv, args.data_dir)