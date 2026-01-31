import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm
import lidar_utils  # Ton fichier utilitaire habituel

# Configuration des dossiers de sortie
OUTPUT_DIR = "dataset_prepared"
# Rappel des classes pour créer les dossiers
CLASS_NAMES = {0: "Antenna", 1: "Cable", 2: "Pole", 3: "Turbine"}


def extract_objects(csv_path, data_root_dir):
    # 1. Charger le Gros CSV (Train_Labels.csv)
    print(f"📂 Lecture du CSV : {csv_path}")
    try:
        df_labels = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Erreur lecture CSV : {e}")
        return

    # Vérification que la colonne 'file_source' existe bien (créée par step1_batch)
    if 'file_source' not in df_labels.columns:
        print("❌ ERREUR CRITIQUE : La colonne 'file_source' est absente du CSV.")
        print("   Avez-vous bien utilisé 'step1_batch.py' ?")
        return

    # 2. Créer l'arborescence : dataset_prepared/Antenna, dataset_prepared/Pole, etc.
    for class_name in CLASS_NAMES.values():
        folder_path = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(folder_path, exist_ok=True)

    # 3. Grouper par fichier source pour optimiser (on ouvre le fichier .h5 une seule fois)
    grouped = df_labels.groupby('file_source')

    print(f"🚀 Début de l'extraction sur {len(grouped)} fichiers sources distincts...")

    total_extracted = 0
    skipped_empty = 0

    # Barre de progression sur les fichiers
    for filename, group in tqdm(grouped, desc="Extraction"):

        # Reconstruire le chemin complet du fichier H5
        file_path = os.path.join(data_root_dir, filename)

        if not os.path.exists(file_path):
            # Parfois le CSV a des noms relatifs, on essaie de trouver le fichier
            print(f"⚠️ Fichier introuvable : {file_path} (Ignoré)")
            continue

        # --- CHARGEMENT DU FICHIER LIDAR (Une seule fois pour tout le groupe) ---
        try:
            df_lidar = lidar_utils.load_h5_data(file_path)
            # On ne garde que les coord XYZ (et on nettoie les points à distance 0)
            df_lidar = df_lidar[df_lidar["distance_cm"] > 0]
            xyz = lidar_utils.spherical_to_local_cartesian(df_lidar)
        except Exception as e:
            print(f"❌ Erreur lecture H5 {filename}: {e}")
            continue

        # --- EXTRACTION DES OBJETS ---
        for idx, row in group.iterrows():
            # Récupérer la boîte
            cx = row['bbox_center_x']
            cy = row['bbox_center_y']
            cz = row['bbox_center_z']

            dx = row['bbox_width']
            dy = row['bbox_length']
            dz = row['bbox_height']

            label = row['class_label']

            # Définir les limites (Min/Max)
            min_x, max_x = cx - dx / 2, cx + dx / 2
            min_y, max_y = cy - dy / 2, cy + dy / 2
            min_z, max_z = cz - dz / 2, cz + dz / 2

            # Masque numpy rapide : Garder uniquement les points DANS la boîte
            mask = (
                    (xyz[:, 0] >= min_x) & (xyz[:, 0] <= max_x) &
                    (xyz[:, 1] >= min_y) & (xyz[:, 1] <= max_y) &
                    (xyz[:, 2] >= min_z) & (xyz[:, 2] <= max_z)
            )

            points_inside = xyz[mask]

            # Sécurité : Si la boîte est vide ou presque (bruit résiduel)
            if len(points_inside) < 5:
                skipped_empty += 1
                continue

            # --- NORMALISATION (IMPORTANT POUR L'IA) ---
            # On centre l'objet sur (0,0,0) pour que l'IA apprenne la forme locale
            points_inside[:, 0] -= cx
            points_inside[:, 1] -= cy
            points_inside[:, 2] -= cz

            # --- SAUVEGARDE ---
            # Nom unique : Label + ID_unique_du_CSV.npy
            # Exemple : Pole_4521.npy
            save_name = f"{label}_{idx}.npy"
            save_path = os.path.join(OUTPUT_DIR, label, save_name)

            # Sauvegarde en binaire optimisé
            np.save(save_path, points_inside)
            total_extracted += 1

    print(f"\n✅ TERMINÉ !")
    print(f"📊 Objets extraits avec succès : {total_extracted}")
    print(f"🗑️  Objets vides ignorés : {skipped_empty}")
    print(f"📁 Dossier de sortie : {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Chemin vers Train_Labels.csv")
    parser.add_argument("--data_dir", required=True, help="Dossier contenant les fichiers .h5 originaux")
    args = parser.parse_args()

    extract_objects(args.csv, args.data_dir)