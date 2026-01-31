import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNet(nn.Module):
    def __init__(self, num_classes=4):
        super(PointNet, self).__init__()

        # --- PARTIE 1 : Extraction de caractéristiques (Point Features) ---
        # On utilise Conv1d car mathématiquement, appliquer un MLP sur chaque point
        # revient à faire une convolution de taille 1 sur le vecteur des points.

        # Entrée : 3 canaux (x, y, z) -> Sortie : 64 caractéristiques
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        # 64 -> 128
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        # 128 -> 1024 (On augmente la dimension pour capturer des formes complexes)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        # --- PARTIE 2 : Classification (Global Features) ---
        # Après le Max Pooling, on aura un vecteur de taille 1024 qui décrit tout l'objet.
        # On le fait passer dans des couches classiques (Fully Connected).

        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, num_classes)

        # Dropout : Pour éviter que le réseau apprenne par cœur (Overfitting)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # x shape en entrée : [Batch_Size, 3, 1024]

        # 1. Extraction locale
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # Ici x shape : [Batch, 1024, 1024]

        # 2. Max Pooling (Global Feature)
        # On prend le max sur la dimension des points (dim 2)
        # C'est l'étape clé qui rend le réseau insensible à l'ordre des points.
        x = torch.max(x, 2, keepdim=True)[0]
        # Ici x shape : [Batch, 1024, 1]

        x = x.view(-1, 1024)  # On aplatit pour les couches FC : [Batch, 1024]

        # 3. Classification MLP
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)

        x = self.fc3(x)  # Sortie finale : [Batch, num_classes]

        return x  # On retourne les "logits" (scores bruts avant probabilité)


if __name__ == "__main__":
    # Petit test pour vérifier que les dimensions collent
    simulated_input = torch.rand(10, 3, 1024)  # 10 objets, 3 coords, 1024 points
    model = PointNet(num_classes=4)
    output = model(simulated_input)
    print(f"✅ Test Modèle OK !")
    print(f"Entrée : {simulated_input.shape}")
    print(f"Sortie : {output.shape} (Doit être [10, 4])")