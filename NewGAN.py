import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# -------------------------------------------------------------------------
# 1) Načtení MNIST z idx souborů
# -------------------------------------------------------------------------
def load_mnist_images_labels(images_path, labels_path):
    with open(images_path, 'rb') as imgpath, open(labels_path, 'rb') as lblpath:
        imgpath.read(16)
        lblpath.read(8)
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
        image = self.images[idx].reshape(28, 28).astype(np.float32)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# -------------------------------------------------------------------------
# 2) Hinge loss bez label smoothingu
# -------------------------------------------------------------------------
def discriminator_hinge_loss(real_logits, fake_logits):
    loss_real = torch.mean(torch.relu(1.0 - real_logits))
    loss_fake = torch.mean(torch.relu(1.0 + fake_logits))
    return loss_real + loss_fake

def generator_hinge_loss(fake_logits):
    return -torch.mean(fake_logits)

# -------------------------------------------------------------------------
# 3) R1 penalizace na reálných vzorcích
# -------------------------------------------------------------------------
def r1_gradient_penalty(real_imgs, real_logits):
    grad_real = torch.autograd.grad(
        outputs=real_logits.sum(),
        inputs=real_imgs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    grad_pen = torch.mean(grad_real.pow(2).sum(dim=[1,2,3]))
    return grad_pen

# -------------------------------------------------------------------------
# 4) Klasický DCGAN-styl architektury (bez další vrstvy)
# -------------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim=100, base_ch=64):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, base_ch*8*7*7),
            nn.BatchNorm1d(base_ch*8*7*7),
            nn.ReLU(True)
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch*8, base_ch*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch*4),
            nn.ReLU(True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch*2),
            nn.ReLU(True)
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(base_ch*2, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 8*64, 7, 7)
        x = self.deconv1(x)
        x = self.deconv2(x)
        img = self.conv_out(x)
        return img

class Discriminator(nn.Module):
    def __init__(self, base_ch=64):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(1, base_ch, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_ch, base_ch*2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc = nn.Linear(base_ch*2 * 7 * 7, 1)

    def forward(self, img):
        x = self.conv1(img)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        validity = self.fc(x)
        return validity

# -------------------------------------------------------------------------
# 5) Trénovací smyčka s n_critic=1 a r1_weight=1.0, lr_d=lr_g=2e-4, 25 epoch
# -------------------------------------------------------------------------
def main():
    # Cesty k MNIST idx souborům
    data_path = r'C:/Users/uzivatel 1/OneDrive/Dokumenty/Coding Projects/Bachelor_Thesis/MNIST_data/raw'
    train_images_path = os.path.join(data_path, 'train-images-idx3-ubyte')
    train_labels_path = os.path.join(data_path, 'train-labels-idx1-ubyte')
    images, labels = load_mnist_images_labels(train_images_path, train_labels_path)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = MNISTDataset(images, labels, transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)

    # Cesty pro ukládání
    output_path = r'C:/Users/uzivatel 1/OneDrive/Dokumenty/Coding Projects/Bachelor_Thesis/generated_images'
    os.makedirs(output_path, exist_ok=True)
    model_path = r'C:/Users/uzivatel 1/OneDrive/Dokumenty/Coding Projects/Bachelor_Thesis/saved_models'
    os.makedirs(model_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Zařízení:", device)

    # Hyperparametry
    latent_dim = 100
    base_ch = 64
    lr = 2e-4
    n_critic = 1
    r1_weight = 1.0
    num_epochs = 25

    G = Generator(latent_dim, base_ch=base_ch).to(device)
    D = Discriminator(base_ch=base_ch).to(device)

    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.9))

    print("Start tréninku...")
    for epoch in range(num_epochs):
        for i, (real_imgs, _) in enumerate(dataloader):
            batch_size_curr = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            # --- Trénink D ---
            real_imgs.requires_grad_(True)
            optimizer_D.zero_grad()

            z = torch.randn(batch_size_curr, latent_dim, device=device)
            fake_imgs = G(z).detach()

            real_logits = D(real_imgs)
            fake_logits = D(fake_imgs)

            loss_D = discriminator_hinge_loss(real_logits, fake_logits)
            gp = r1_gradient_penalty(real_imgs, real_logits)
            loss_D_total = loss_D + r1_weight * gp

            loss_D_total.backward()
            optimizer_D.step()
            real_imgs.requires_grad_(False)

            # --- Trénink G ---
            z = torch.randn(batch_size_curr, latent_dim, device=device)
            fake_imgs = G(z)
            optimizer_G.zero_grad()
            fake_logits = D(fake_imgs)
            loss_G = generator_hinge_loss(fake_logits)
            loss_G.backward()
            optimizer_G.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch {i}/{len(dataloader)} "
                      f"Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}, GP: {gp.item():.4f}")

        # Uložení ukázkových obrázků
        with torch.no_grad():
            z = torch.randn(64, latent_dim, device=device)
            gen_imgs = G(z)
            gen_imgs = (gen_imgs + 1) / 2
            img_file = os.path.join(output_path, f"epoch_{epoch+1}.png")
            utils.save_image(gen_imgs, img_file, nrow=8)
            print(f"Uloženy obrázky: {img_file}")

        # Uložení modelů
        torch.save(G.state_dict(), os.path.join(model_path, f"generator_epoch_{epoch+1}.pth"))
        torch.save(D.state_dict(), os.path.join(model_path, f"discriminator_epoch_{epoch+1}.pth"))
        print(f"Modely uloženy pro epochu {epoch+1}")

    print("Trénink dokončen.")

if __name__ == "__main__":
    main()
