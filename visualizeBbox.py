import argparse
import numpy as np
import open3d as o3d
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import Normalize
import lidar_utils
import os

window_width, window_height = 1280, 720

# --- COULEURS DES CLASSES (Pour les boîtes) ---
BOX_COLORS = {
    0: [0, 0, 1],  # Antenna (Bleu)
    1: [1, 0.8, 0],  # Cable (Jaune/Or)
    2: [1, 0, 0],  # Pole (Rouge)
    3: [0, 1, 0]  # Turbine (Vert)
}


def clean_lidar_points(df):
    """Nettoie les points invalides (distance=0)."""
    df = df[df["distance_cm"] > 0]
    return df.reset_index(drop=True)


def get_lineset_from_bbox(row):
    """
    Crée une boîte filaire 3D (LineSet) à partir d'une ligne du CSV.
    Gère les colonnes 'center_x' ou 'bbox_center_x' selon ta version du CSV.
    """
    # 1. Gestion des noms de colonnes (au cas où tu as l'ancienne ou la nouvelle version)
    cx = row['center_x'] if 'center_x' in row else row['bbox_center_x']
    cy = row['center_y'] if 'center_y' in row else row['bbox_center_y']
    cz = row['center_z'] if 'center_z' in row else row['bbox_center_z']

    dx = row['size_x'] if 'size_x' in row else row['bbox_width']
    dy = row['size_y'] if 'size_y' in row else row['bbox_length']
    dz = row['size_z'] if 'size_z' in row else row['bbox_height']

    # 2. Calcul des 8 coins
    x0, x1 = cx - dx / 2, cx + dx / 2
    y0, y1 = cy - dy / 2, cy + dy / 2
    z0, z1 = cz - dz / 2, cz + dz / 2

    points = [
        [x0, y0, z0], [x1, y0, z0], [x0, y1, z0], [x1, y1, z0],  # Bas
        [x0, y0, z1], [x1, y0, z1], [x0, y1, z1], [x1, y1, z1]  # Haut
    ]

    # 3. Lignes reliant les coins
    lines = [
        [0, 1], [0, 2], [1, 3], [2, 3],  # Face du bas
        [4, 5], [4, 6], [5, 7], [6, 7],  # Face du haut
        [0, 4], [1, 5], [2, 6], [3, 7]  # Piliers verticaux
    ]

    # 4. Création de l'objet Open3D
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Couleur selon la classe ID
    class_id = int(row['class_id'])
    color = BOX_COLORS.get(class_id, [1, 1, 1])  # Blanc par défaut si inconnu
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])

    return line_set


def main():
    parser = argparse.ArgumentParser(description="Visualize Lidar points + CSV Bounding Boxes")
    parser.add_argument("--file", required=True, help="Path to HDF5 lidar file")
    parser.add_argument("--csv", required=False, help="Path to the generated CSV file (optional)")
    parser.add_argument("--pose-index", type=int, default=0, help="Index of the pose to visualize")
    parser.add_argument("--cmap", default="turbo", help="Colormap for intensity")

    args = parser.parse_args()

    # --- 1. Chargement du Nuage de Points (Ton code qui marche) ---
    try:
        df = lidar_utils.load_h5_data(args.file)
        print(f"Chargé : {len(df)} points depuis {args.file}")
    except Exception as e:
        print(f"Erreur chargement H5: {e}")
        return

    if len(df) == 0: return

    # Nettoyage
    df = clean_lidar_points(df)

    # Gestion des Poses
    pose_counts = lidar_utils.get_unique_poses(df)

    if args.pose_index >= len(pose_counts):
        print(f"Erreur: L'index {args.pose_index} n'existe pas. Max = {len(pose_counts) - 1}")
        return

    # Sélection de la frame
    selected_pose = pose_counts.iloc[args.pose_index]
    df_frame = lidar_utils.filter_by_pose(df, selected_pose)
    print(f"Visualisation Frame {args.pose_index} ({len(df_frame)} points)")

    # Conversion XYZ
    xyz = lidar_utils.spherical_to_local_cartesian(df_frame)

    # Création du Nuage Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # Coloriage
    if {"r", "g", "b"}.issubset(df_frame.columns):
        print("-> Mode Couleur : Vérité Terrain RGB")
        rgb = np.column_stack((
            df_frame["r"].to_numpy() / 255.0,
            df_frame["g"].to_numpy() / 255.0,
            df_frame["b"].to_numpy() / 255.0
        ))
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    else:
        print("-> Mode Couleur : Réflectivité")
        intensities = df_frame["reflectivity"].to_numpy()
        norm = Normalize(vmin=intensities.min(), vmax=intensities.max())
        cmap = colormaps.get_cmap(args.cmap)
        colors = cmap(norm(intensities))[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # Liste des objets à afficher (On commence par le nuage)
    geometries = [pcd]

    # --- 2. Chargement et Ajout des Boîtes (Nouvelle partie) ---
    if args.csv and os.path.exists(args.csv):
        print(f"Chargement des boîtes depuis : {args.csv}")
        try:
            df_bbox = pd.read_csv(args.csv)

            # Filtre : On ne garde que les boîtes de la frame actuelle
            # (On suppose que la colonne s'appelle 'frame_idx' ou 'pose_index')
            col_frame = 'frame_idx' if 'frame_idx' in df_bbox.columns else 'pose_index'
            current_boxes = df_bbox[df_bbox[col_frame] == args.pose_index]

            print(f"-> {len(current_boxes)} boîtes trouvées pour la Frame {args.pose_index}")

            for _, row in current_boxes.iterrows():
                bbox_line = get_lineset_from_bbox(row)
                geometries.append(bbox_line)

        except Exception as e:
            print(f"⚠️ Erreur lors de la lecture du CSV : {e}")
    elif args.csv:
        print(f"⚠️ Fichier CSV introuvable : {args.csv}")

    # --- 3. Affichage ---
    print("Lancement de la fenêtre 3D...")

    # Configuration Caméra
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Frame {args.pose_index}", width=window_width, height=window_height)

    for geom in geometries:
        vis.add_geometry(geom)

    ctrl = vis.get_view_control()
    cam_pos = np.array([0.0, 0.0, 0.0])
    forward = np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    lookat = cam_pos + 20.0 * forward

    ctrl.set_lookat(lookat)
    ctrl.set_front(-forward)
    ctrl.set_up(up)
    ctrl.set_zoom(0.1)

    # Options de rendu (Points plus gros pour mieux voir)
    opt = vis.get_render_option()
    opt.point_size = 2.0
    #opt.point_size = 3.0
    #opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Fond gris foncé

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()