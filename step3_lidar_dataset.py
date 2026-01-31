import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class LidarDataset(Dataset):
    def __init__(self, data_dir, num_points=1024, partition='train', augmentation=True):
        """
        data_dir: Dossier 'dataset_prepared'
        num_points: Nombre de points fixe imposé à l'IA (Standard = 1024)
        partition: 'train' ou 'test' (ici on utilise tout pour train pour l'instant)
        """
        self.data_dir = data_dir
        self.num_points = num_points
        self.augmentation = augmentation

        self.classes = {"Antenna": 0, "Cable": 1, "Pole": 2, "Turbine": 3}
        self.files = []
        self.labels = []

        # Chargement de la liste des fichiers
        for class_name, class_id in self.classes.items():
            path = os.path.join(data_dir, class_name, "*.npy")
            class_files = glob.glob(path)
            self.files.extend(class_files)
            self.labels.extend([class_id] * len(class_files))

        print(f"📦 Dataset chargé : {len(self.files)} objets trouvés.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 1. Lire le fichier .npy
        file_path = self.files[idx]
        label = self.labels[idx]

        try:
            point_set = np.load(file_path).astype(np.float32)
        except:
            # Sécurité si un fichier est corrompu, on renvoie des zéros
            return torch.zeros((3, self.num_points)), torch.tensor(label)

        # 2. Échantillonnage (Sampling) pour avoir exactement num_points
        # C'est CRUCIAL : PointNet veut une matrice fixe.
        choice = np.random.choice(len(point_set), self.num_points, replace=True)
        point_set = point_set[choice, :]

        # 3. Data Augmentation (Rendre l'IA robuste)
        if self.augmentation:
            # Rotation aléatoire autour de Z (l'objet peut être vu sous n'importe quel angle)
            theta = np.random.uniform(0, 2 * np.pi)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            point_set = np.dot(point_set, rotation_matrix)

            # Jitter (Ajouter un micro-bruit pour simuler l'imprécision du lidar)
            point_set += np.random.normal(0, 0.02, size=point_set.shape)

        # 4. Normalisation finale (Centrer dans une sphère unitaire)
        point_set = point_set - np.mean(point_set, axis=0)
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)))
        if dist > 0:  # Eviter division par zero
            point_set = point_set / dist

        # 5. Format PyTorch : (3, N) au lieu de (N, 3)
        # PointNet aime avoir les Canaux (XYZ) en premier
        point_set = point_set.transpose(1, 0)

        return torch.from_numpy(point_set), torch.tensor(label, dtype=torch.long)


if __name__ == "__main__":
    # Petit test pour vérifier que ça marche
    ds = LidarDataset("dataset_prepared", augmentation=False)
    print(f"Exemple d'item shape: {ds[0][0].shape}")  # Devrait afficher torch.Size([3, 1024])