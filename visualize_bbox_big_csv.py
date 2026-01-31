import argparse
import numpy as np
import open3d as o3d
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import Normalize
import lidar_utils
import os

window_width, window_height = 1280, 720

# --- COULEURS DES CLASSES ---
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
    """Crée une boîte filaire 3D (LineSet) à partir d'une ligne du CSV."""
    # 1. Gestion des noms de colonnes (Compatibilité ancien/nouveau CSV)
    cx = row.get('bbox_center_x', row.get('center_x'))
    cy = row.get('bbox_center_y', row.get('center_y'))
    cz = row.get('bbox_center_z', row.get('center_z'))

    dx = row.get('bbox_width', row.get('size_x'))
    dy = row.get('bbox_length', row.get('size_y'))
    dz = row.get('bbox_height', row.get('size_z'))

    # 2. Calcul des 8 coins
    x0, x1 = cx - dx / 2, cx + dx / 2
    y0, y1 = cy - dy / 2, cy + dy / 2
    z0, z1 = cz - dz / 2, cz + dz / 2

    points = [
        [x0, y0, z0], [x1, y0, z0], [x0, y1, z0], [x1, y1, z0],  # Bas
        [x0, y0, z1], [x1, y0, z1], [x0, y1, z1], [x1, y1, z1]  # Haut
    ]

    lines = [
        [0, 1], [0, 2], [1, 3], [2, 3],  # Bas
        [4, 5], [4, 6], [5, 7], [6, 7],  # Haut
        [0, 4], [1, 5], [2, 6], [3, 7]  # Piliers
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    class_id = int(row['class_id'])
    color = BOX_COLORS.get(class_id, [1, 1, 1])
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])

    return line_set


def main():
    parser = argparse.ArgumentParser(description="Visualize Lidar + Global CSV Bounding Boxes")
    parser.add_argument("--file", required=True, help="Path to HDF5 lidar file (ex: data/scene_100.h5)")
    parser.add_argument("--csv", required=False, help="Path to the BIG generated CSV file")
    parser.add_argument("--pose-index", type=int, default=0, help="Index of the pose to visualize")
    parser.add_argument("--cmap", default="turbo", help="Colormap for intensity")

    args = parser.parse_args()

    # --- 1. Récupérer le nom du fichier pur ---
    # Si args.file vaut "C:/Data/Train/File_ABC.h5", on garde "File_ABC.h5"
    current_h5_filename = os.path.basename(args.file)

    # --- 2. Chargement du Nuage de Points ---
    try:
        df = lidar_utils.load_h5_data(args.file)
        print(f"✅ Chargé : {len(df)} points depuis {args.file}")
    except Exception as e:
        print(f"❌ Erreur chargement H5: {e}")
        return

    if len(df) == 0: return

    df = clean_lidar_points(df)
    pose_counts = lidar_utils.get_unique_poses(df)

    if args.pose_index >= len(pose_counts):
        print(f"⚠️ Erreur: L'index {args.pose_index} n'existe pas.")
        return

    selected_pose = pose_counts.iloc[args.pose_index]
    df_frame = lidar_utils.filter_by_pose(df, selected_pose)
    print(f"Visualisation : {current_h5_filename} | Frame {args.pose_index} ({len(df_frame)} pts)")

    xyz = lidar_utils.spherical_to_local_cartesian(df_frame)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # Couleurs
    if {"r", "g", "b"}.issubset(df_frame.columns):
        rgb = np.column_stack((df_frame["r"] / 255.0, df_frame["g"] / 255.0, df_frame["b"] / 255.0))
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    else:
        intensities = df_frame["reflectivity"].to_numpy()
        norm = Normalize(vmin=intensities.min(), vmax=intensities.max())
        cmap = colormaps.get_cmap(args.cmap)
        colors = cmap(norm(intensities))[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)

    geometries = [pcd]

    # --- 3. Chargement Intelligent des Boîtes ---
    if args.csv and os.path.exists(args.csv):
        print(f"📂 Lecture du CSV : {args.csv}")
        try:
            df_bbox = pd.read_csv(args.csv)

            # --- FILTRE 1 : Sélectionner le bon fichier source ---
            # On vérifie si la colonne 'file_source' existe (créée par step1_batch)
            if 'file_source' in df_bbox.columns:
                df_bbox = df_bbox[df_bbox['file_source'] == current_h5_filename]
                print(f"   -> Filtre fichier : {current_h5_filename}")
            else:
                print("   ⚠️ Attention : Colonne 'file_source' absente. Affichage risqué (mélange possible).")

            # --- FILTRE 2 : Sélectionner la bonne frame ---
            col_frame = 'pose_index' if 'pose_index' in df_bbox.columns else 'frame_idx'
            current_boxes = df_bbox[df_bbox[col_frame] == args.pose_index]

            print(f"   -> {len(current_boxes)} boîtes trouvées pour cette frame.")

            for _, row in current_boxes.iterrows():
                bbox_line = get_lineset_from_bbox(row)
                geometries.append(bbox_line)

        except Exception as e:
            print(f"⚠️ Erreur CSV : {e}")

    # --- 4. Affichage ---
    print("🖥️  Lancement de la fenêtre 3D...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"{current_h5_filename} - Frame {args.pose_index}", width=window_width,
                      height=window_height)

    for geom in geometries:
        vis.add_geometry(geom)

    # Réglages caméra
    ctrl = vis.get_view_control()

    cam_pos = np.array([0.0, 0.0, 0.0])
    forward = np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    lookat = cam_pos + 20.0 * forward

    ctrl.set_lookat(lookat)
    ctrl.set_front(-forward)
    ctrl.set_up(up)
    ctrl.set_zoom(0.1)

    opt = vis.get_render_option()
    opt.point_size = 2.0

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()