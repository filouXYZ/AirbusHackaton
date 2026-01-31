import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os

# Import de tes scripts
from step3_lidar_dataset import LidarDataset
from step3_2_model import PointNet

# --- CONFIGURATION (HYPERPARAMÈTRES) ---
DATA_DIR = "dataset_prepared"
BATCH_SIZE = 32  # Nombre d'objets traités d'un coup
EPOCHS = 20  # Nombre de fois qu'on voit tout le dataset
LEARNING_RATE = 0.001  # Vitesse d'apprentissage
NUM_POINTS = 1024  # Points par objet (doit matcher lidar_dataset)
CLASSES = 4  # Antenna, Cable, Pole, Turbine


def train():
    # 1. Configuration du Matériel (GPU RTX is King 👑)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Démarrage de l'entraînement sur : {device}")
    if device.type == 'cuda':
        print(f"   Carte : {torch.cuda.get_device_name(0)}")

    # 2. Préparation des Données
    print("📦 Chargement du Dataset...")
    full_dataset = LidarDataset(DATA_DIR, num_points=NUM_POINTS, augmentation=True)

    # On coupe : 80% pour s'entraîner, 20% pour vérifier (Validation)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"📊 Train set: {len(train_dataset)} objets | Val set: {len(val_dataset)} objets")

    # 3. Initialisation du Modèle
    model = PointNet(num_classes=CLASSES).to(device)

    # L'Optimiseur (Adam est le standard, il gère la vitesse d'apprentissage tout seul)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # La Fonction de Coût (CrossEntropy pour la classification multiple)
    criterion = nn.CrossEntropyLoss()

    # 4. Boucle d'Entraînement
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()  # Mode Entraînement (Active Dropout, BatchNorm...)
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        # --- PHASE D'APPRENTISSAGE ---
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")
        for points, labels in loop:
            points, labels = points.to(device), labels.to(device)

            # A. Remise à zéro des gradients
            optimizer.zero_grad()

            # B. Forward (Prédiction)
            outputs = model(points)

            # C. Calcul de l'erreur (Loss)
            loss = criterion(outputs, labels)

            # D. Backward (Correction des neurones)
            loss.backward()
            optimizer.step()

            # Stats
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Mise à jour barre de progression
            loop.set_postfix(loss=loss.item())

        train_acc = 100 * correct_train / total_train

        # --- PHASE DE VALIDATION (Examens blancs) ---
        model.eval()  # Mode Évaluation (Fige le modèle)
        correct_val = 0
        total_val = 0

        with torch.no_grad():  # Pas de calcul de gradient ici (économie RAM)
            for points, labels in val_loader:
                points, labels = points.to(device), labels.to(device)
                outputs = model(points)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_acc = 100 * correct_val / total_val

        # --- RAPPORT FIN D'EPOCH ---
        print(f"🏁 Epoch {epoch + 1}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # Sauvegarde si c'est le meilleur score
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"💾 Nouveau record ! Modèle sauvegardé (Acc: {best_acc:.2f}%)")

        print("-" * 50)

    print("🎉 Entraînement terminé !")


if __name__ == "__main__":
    train()