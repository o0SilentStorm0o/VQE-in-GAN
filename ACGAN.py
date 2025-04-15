import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image

# Nastavení seed pro reprodukovatelnost
torch.manual_seed(42)
np.random.seed(42)

# Cesta k MNIST datasetu
MNIST_PATH = "C:/Users/uzivatel 1/OneDrive/Dokumenty/Coding Projects/Bachelor_Thesis/MNIST_data/raw"

# Hyperparametry
batch_size = 64
z_dim = 100  # Dimenze latentního prostoru
num_classes = 10  # Počet tříd v MNIST (0-9)
img_channels = 1  # MNIST má jen jeden kanál (černobílé obrázky)
learning_rate = 0.0002
betas = (0.5, 0.999)
num_epochs = 50

# Zařízení (GPU pokud je dostupné)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Použité zařízení: {device}")

# Datové transformace
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalizace do rozsahu [-1, 1]
])

# Načtení datasetu
try:
    train_dataset = datasets.MNIST(root=MNIST_PATH, train=True, transform=transform, download=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset úspěšně načten: {len(train_dataset)} obrázků")
except Exception as e:
    print(f"Nepodařilo se načíst dataset: {e}")
    print("Zkusím stáhnout dataset...")
    train_dataset = datasets.MNIST(root=MNIST_PATH, train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset úspěšně stažen: {len(train_dataset)} obrázků")

# Definice Generátoru (G)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # Latentní vektor z (100) + embedding třídy (10) -> 110
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        # Vstupní lineární vrstva pro zpracování spojení z a embedovaného labelu
        self.input = nn.Linear(z_dim + num_classes, 256 * 7 * 7)
        
        # Hlavní část generátoru
        self.main = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2),  # 7x7 -> 14x14
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # 14x14 -> 28x28
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # Embedování labelu
        label_embedding = self.label_embedding(labels)
        
        # Spojení latentního vektoru a labelů
        z = torch.cat([z, label_embedding], dim=1)
        
        # Zpracování přes lineární vrstvu a reshape
        x = self.input(z)
        x = x.view(-1, 256, 7, 7)
        
        # Průchod hlavní částí generátoru
        return self.main(x)

# Definice Diskriminátoru (D) s duálními výstupy
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Hlavní část diskriminátoru
        self.main = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(img_channels, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),

            # 14x14 -> 7x7
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),

            # 7x7 -> 4x4  <-- ZMENA TU
            nn.Conv2d(128, 256, 4, stride=1, padding=0), # Zmena kernel_size z 3 na 4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
        )

        # Výstup pro rozlišení pravý/falešný
        self.source_output = nn.Linear(256 * 4 * 4, 1) 

        # Výstup pro klasifikaci třídy
        self.class_output = nn.Linear(256 * 4 * 4, num_classes)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Průchod hlavní částí
        x = self.main(x)
        x = x.view(-1, 256 * 4 * 4)
        
        # Dva výstupy: pravděpodobnost pravosti a klasifikace třídy
        validity = self.sigmoid(self.source_output(x))
        class_logits = self.class_output(x) # Priamo logity
        return validity, class_logits

# Inicializace modelů
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Funkce pro váhovou inicializaci
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Aplikace inicializace vah
generator.apply(weights_init)
discriminator.apply(weights_init)

# Definice loss funkcí
adversarial_loss = nn.BCELoss()
auxiliary_loss = nn.CrossEntropyLoss()

# Optimizéry
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=betas)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=betas)

# Funkce pro generování a ukládání vzorků
def save_samples(epoch, fixed_noise, fixed_labels):
    generator.eval()
    with torch.no_grad():
        gen_imgs = generator(fixed_noise, fixed_labels)
        
    # Vytvoření mřížky obrázků pro každou třídu
    fig, axs = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(10):
        row, col = i // 5, i % 5
        axs[row, col].imshow(gen_imgs[i].cpu().detach().squeeze(), cmap='gray')
        axs[row, col].set_title(f"Třída: {fixed_labels[i].item()}")
        axs[row, col].axis('off')
    
    plt.tight_layout()
    os.makedirs("images", exist_ok=True)
    plt.savefig(f"images/acgan_mnist_epoch_{epoch}.png")
    plt.close()
    
    # Uložení mřížky všech obrázků
    save_image(gen_imgs.data, f"images/acgan_samples_epoch_{epoch}.png", 
              nrow=5, normalize=True)
    generator.train()

# Příprava pevného šumu a labelů pro vizualizaci během trénování
fixed_noise = torch.randn(10, z_dim, device=device)
fixed_labels = torch.LongTensor(np.arange(10)).to(device)

# Tréninkový cyklus
for epoch in range(num_epochs):
    for i, (real_imgs, labels) in enumerate(train_loader):
        batch_size = real_imgs.size(0)
        
        # Příprava labelů
        real_labels = torch.ones(batch_size, 1).to(device)  # 1 pro pravé
        fake_labels = torch.zeros(batch_size, 1).to(device)  # 0 pro falešné
        
        # Přesun dat na zařízení
        real_imgs = real_imgs.to(device)
        labels = labels.to(device)
        
        # -----------------
        # Trénink Diskriminátoru
        # -----------------
        optimizer_D.zero_grad()
        
        # Ohodnocení pravých obrázků
        validity_real, class_probs_real = discriminator(real_imgs)
        
        # Loss pro pravé obrázky (adversarial + auxiliary)
        d_real_loss = adversarial_loss(validity_real, real_labels)
        d_real_aux_loss = auxiliary_loss(class_probs_real, labels)
        
        # Generování falešných obrázků
        z = torch.randn(batch_size, z_dim).to(device)
        gen_labels = torch.randint(0, num_classes, (batch_size,)).to(device)
        gen_imgs = generator(z, gen_labels)
        
        # Ohodnocení falešných obrázků
        validity_fake, class_probs_fake = discriminator(gen_imgs.detach())
        
        # Loss pro falešné obrázky (adversarial + auxiliary)
        d_fake_loss = adversarial_loss(validity_fake, fake_labels)
        d_fake_aux_loss = auxiliary_loss(class_probs_fake, gen_labels)
        
        # Celkový loss diskriminátoru
        d_loss = d_real_loss + d_fake_loss + (d_real_aux_loss + d_fake_aux_loss)
        
        d_loss.backward()
        optimizer_D.step()
        
        # -----------------
        # Trénink Generátoru
        # -----------------
        optimizer_G.zero_grad()
        
        # Klasifikace generovaných obrázků diskriminátorem
        validity, class_probs = discriminator(gen_imgs)
        
        # Loss generátoru (adversarial + auxiliary)
        g_loss = adversarial_loss(validity, real_labels) + auxiliary_loss(class_probs, gen_labels)
        
        g_loss.backward()
        optimizer_G.step()
        
        # Výpis pokroku
        if i % 100 == 0:
            print(
                f"[Epoch {epoch}/{num_epochs}] "
                f"[Batch {i}/{len(train_loader)}] "
                f"[D loss: {d_loss.item():.4f}] "
                f"[G loss: {g_loss.item():.4f}]"
            )
    
    # Generování a ukládání vzorků na konci každé epochy
    save_samples(epoch, fixed_noise, fixed_labels)

# Uložení natrénovaných modelů
torch.save(generator.state_dict(), 'acgan_generator.pth')
torch.save(discriminator.state_dict(), 'acgan_discriminator.pth')
print("Trénink dokončen! Modely uloženy.")

# Generování finálních vzorků
with torch.no_grad():
    # Generování více vzorků pro každou třídu
    rows, cols = 10, 10
    samples_per_class = cols
    
    # Pro každou třídu:
    for digit in range(rows):
        z = torch.randn(samples_per_class, z_dim, device=device)
        labels = torch.LongTensor([digit] * samples_per_class).to(device)
        
        # Generování obrázků
        gen_imgs = generator(z, labels)
        
        # Uložení vygenerovaných obrázků
        save_image(gen_imgs.data, f"images/acgan_digit_{digit}.png", 
                  nrow=cols, normalize=True)
    
    print("Vygenerovány finální vzorky pro každou třídu.")