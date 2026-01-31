import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from step3_lidar_dataset import LidarDataset
from step3_2_model import PointNet

# --- CONFIG ---
DATA_DIR = "dataset_prepared"
MODEL_PATH = "best_model.pth"
CLASSES = ["Antenna", "Cable", "Pole", "Turbine"]


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 Évaluation sur : {device}")

    # 1. Charger les données (Sans augmentation, on veut la vérité brute)
    dataset = LidarDataset(DATA_DIR, partition='test', augmentation=False)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 2. Charger le Cerveau
    model = PointNet(num_classes=len(CLASSES)).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("✅ Modèle chargé avec succès.")
    except FileNotFoundError:
        print("❌ Erreur : 'best_model.pth' introuvable. Lance step4_train.py d'abord !")
        return

    model.eval()
    all_preds = []
    all_labels = []

    # 3. Faire passer l'examen
    print("📝 Calcul des prédictions...")
    with torch.no_grad():
        for points, labels in loader:
            points = points.to(device).float()  # Correction float 32
            labels = labels.to(device)

            outputs = model(points)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Dessiner la Matrice
    cm = confusion_matrix(all_labels, all_preds)

    # Calcul en pourcentage pour que ce soit lisible
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.ylabel('Vraie Classe')
    plt.xlabel('Classe Prédite par IA')
    plt.title('Matrice de Confusion (%)')
    plt.show()


if __name__ == "__main__":
    evaluate()