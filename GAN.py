import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils

# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("Warning: GPU not found. Running on CPU might be very slow!")
else:
    print(f"Using device: {device}")

# Path to your MNIST data
data_path = r'C:/Users/uzivatel 1/OneDrive/Dokumenty/Coding Projects/Bachelor_Thesis/MNIST_data/raw/'
# Path to save generated images
output_path = r'C:/Users/uzivatel 1/OneDrive/Dokumenty/Coding Projects/Bachelor_Thesis/generated_images/'
os.makedirs(output_path, exist_ok=True)

# Path to save trained models
model_path = r'C:/Users/uzivatel 1/OneDrive/Dokumenty/Coding Projects/Bachelor_Thesis/saved_models/'
os.makedirs(model_path, exist_ok=True)

# Definice hyperparametrů
batch_size = 256  # Zvýšení batch size pro lepší využití GPU
learning_rate = 0.0002
epochs = 100

# Vlastní načítání MNIST dat
def load_mnist_images_labels(images_path, labels_path):
    with open(images_path, 'rb') as imgpath, open(labels_path, 'rb') as lblpath:
        # Přečti záhlaví souborů
        imgpath.read(16)  # 16 byte header
        lblpath.read(8)   # 8 byte header
        
        # Načti obrazy a labely
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(-1, 28*28)
        labels = np.frombuffer(lblpath.read(), dtype=np.uint8)
        
        return images, labels

class MNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32) / 255.0  # Normalizace
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return torch.tensor(image).to(device), label

# Načtení dat
train_images, train_labels = load_mnist_images_labels(
    os.path.join(data_path, 'train-images-idx3-ubyte'),
    os.path.join(data_path, 'train-labels-idx1-ubyte')
)

# Dataset a DataLoader
transform = lambda x: (x - 0.5) / 0.5  # Normalizace na -1 až 1 (jako u transforms.Normalize)
mnist_dataset = MNISTDataset(train_images, train_labels, transform=transform)
dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)  # Nastaveno num_workers na 0 pro Windows

# Generátor
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 784),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.main(x)

# Diskriminátor
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.main(x)

# Inicializace generátoru a diskriminátoru
G = Generator().to(device)
D = Discriminator().to(device)

# Ztrátová funkce a optimalizátoři
criterion = nn.BCELoss()
optimizerD = optim.Adam(D.parameters(), lr=learning_rate, betas=(0.5, 0.999))  # Úprava beta hodnot pro stabilnější trénink
optimizerG = optim.Adam(G.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Funkce pro ukládání generovaných obrázků
def save_generated_images(epoch, images, output_path):
    images = images.view(images.size(0), 1, 28, 28)
    grid_img = vutils.make_grid(images, nrow=10, normalize=True)
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(grid_img.cpu().detach().numpy(), (1, 2, 0)), cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(output_path, f'epoch_{epoch}.png'))
    plt.close()

# Trénink
if __name__ == '__main__':
    for epoch in range(epochs):
        for i, (real_data, _) in enumerate(dataloader):
            batch_size = real_data.size(0)

            # Trénink diskriminátoru
            real_data = real_data.view(batch_size, -1).to(device)
            labels_real = torch.ones(batch_size, 1).to(device)
            labels_fake = torch.zeros(batch_size, 1).to(device)

            # Real data loss
            output = D(real_data)
            lossD_real = criterion(output, labels_real)

            # Fake data loss
            noise = torch.randn(batch_size, 100, device=device)
            fake_data = G(noise)
            output = D(fake_data.detach())
            lossD_fake = criterion(output, labels_fake)

            # Celková ztráta diskriminátoru
            lossD = lossD_real + lossD_fake

            D.zero_grad()
            lossD.backward()
            optimizerD.step()

            # Trénink generátoru
            noise = torch.randn(batch_size, 100, device=device)  # Zajisti, že noise se generuje na GPU
            fake_data = G(noise)
            output = D(fake_data)
            lossG = criterion(output, labels_real)  # Generátor se snaží oklamat diskriminátor

            G.zero_grad()
            lossG.backward()
            optimizerG.step()

        # Uložit generované obrázky každých 10 epoch
        if epoch % 10 == 0 or epoch == epochs - 1:
            save_generated_images(epoch, fake_data, output_path)

        print(f"Epoch [{epoch}/{epochs}] Loss D: {lossD.item()}, Loss G: {lossG.item()}")

    # Uložit modely generátoru a diskriminátoru
    torch.save(G.state_dict(), os.path.join(model_path, 'generator.pth'))
    torch.save(D.state_dict(), os.path.join(model_path, 'discriminator.pth'))
