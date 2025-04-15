# Nazev souboru: train_classifier.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# --- Konfigurace ---
MNIST_PATH = "C:/Users/uzivatel 1/OneDrive/Dokumenty/Coding Projects/Bachelor_Thesis/MNIST_data/raw" # Stejná cesta jako v ACGAN.py
CLASSIFIER_SAVE_PATH = "mnist_classifier.pth"
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 10 # Mělo by stačit pro dobrou přesnost na MNIST
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Použité zařízení pro trénink klasifikátoru: {DEVICE}")

# --- Datové transformace (stejné jako v ACGAN pro konzistenci) ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --- Načtení datasetu ---
try:
    train_dataset = datasets.MNIST(root=MNIST_PATH, train=True, transform=transform, download=False)
    test_dataset = datasets.MNIST(root=MNIST_PATH, train=False, transform=transform, download=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("MNIST dataset pro klasifikátor úspěšně načten.")
except Exception as e:
    print(f"Nepodařilo se načíst MNIST dataset: {e}")
    print("Zkusím stáhnout dataset...")
    train_dataset = datasets.MNIST(root=MNIST_PATH, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=MNIST_PATH, train=False, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("MNIST dataset úspěšně stažen.")

# --- Definice jednoduché CNN pro klasifikaci MNIST ---
class MnistClassifier(nn.Module):
    def __init__(self):
        super(MnistClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2) # 28x28 -> 28x28
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2) # 14x14 -> 14x14
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 14x14 -> 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 10) # 10 tříd pro MNIST

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x # Vrátíme logity, CrossEntropyLoss má softmax v sobě

# --- Inicializace modelu, loss funkce a optimizeru ---
classifier = MnistClassifier().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)

# --- Tréninkový cyklus ---
print("Zahajuji trénink klasifikátoru...")
classifier.train()
for epoch in range(EPOCHS):
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = classifier(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if batch_idx % 200 == 0:
             print(f"\tTrénink Epoch: {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
    print(f"Epoch {epoch+1} Průměrná Loss: {epoch_loss / len(train_loader):.4f}")

# --- Testování modelu ---
print("Testuji klasifikátor...")
classifier.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = classifier(data)
        test_loss += criterion(output, target).item()  # Sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
accuracy = 100. * correct / len(test_loader.dataset)
print(f"\nTestovací sada: Průměrná loss: {test_loss:.4f}, Přesnost: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")

# --- Uložení natrénovaného modelu ---
torch.save(classifier.state_dict(), CLASSIFIER_SAVE_PATH)
print(f"Model klasifikátoru uložen do: {CLASSIFIER_SAVE_PATH}")