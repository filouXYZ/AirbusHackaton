import torch
import random
import open3d as o3d
import numpy as np
from step3_lidar_dataset import LidarDataset
from step3_2_model import PointNet

# --- CONFIG ---
DATA_DIR = "dataset_prepared"
MODEL_PATH = "best_model.pth"
CLASSES = {0: "Antenna", 1: "Cable", 2: "Pole", 3: "Turbine"}


def visualize_random():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Charger Dataset et Modèle
    dataset = LidarDataset(DATA_DIR, augmentation=False)
    model = PointNet(num_classes=4).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print("🎲 Pioche d'un objet au hasard...")

    while True:
        # Piocher un index aléatoire
        idx = random.randint(0, len(dataset) - 1)
        points_tensor, label_tensor = dataset[idx]

        # Préparer pour le modèle (Ajouter dimension Batch : [3, 1024] -> [1, 3, 1024])
        input_tensor = points_tensor.unsqueeze(0).to(device).float()

        # Prédiction
        with torch.no_grad():
            output = model(input_tensor)
            _, pred_idx = torch.max(output, 1)

        pred_label = CLASSES[pred_idx.item()]
        true_label = CLASSES[label_tensor.item()]

        # Couleur du texte console
        result = "✅ CORRECT" if pred_label == true_label else "❌ ERREUR"
        print(f"Objet #{idx} | Vrai: {true_label} | IA: {pred_label} -> {result}")

        # --- VISUALISATION 3D ---
        # Remettre les points en format [N, 3] pour Open3D
        xyz = points_tensor.numpy().transpose(1, 0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        # Couleur : Vert si bon, Rouge si mauvais
        color = [0, 1, 0] if pred_label == true_label else [1, 0, 0]
        pcd.paint_uniform_color(color)

        o3d.visualization.draw_geometries(
            [pcd],
            window_name=f"IA: {pred_label} (Vrai: {true_label})",
            width=800, height=600,
            left=50, top=50
        )

        cmd = input("Presse [Enter] pour un autre objet, ou 'q' pour quitter : ")
        if cmd == 'q':
            break


if __name__ == "__main__":
    visualize_random()