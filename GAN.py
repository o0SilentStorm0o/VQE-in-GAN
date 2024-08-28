import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Hyperparameters
    latent_dim = 100
    hidden_dim = 512  # Increased from 256
    image_dim = 28 * 28
    num_epochs = 1000
    batch_size = 512
    lr_g = 0.0005  
    lr_d = 0.0003
    beta1 = 0.5
    beta2 = 0.999

    # Generator Network
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, image_dim),
                nn.Tanh()
            )
        
        def forward(self, z):
            img = self.model(z)
            return img.view(-1, 1, 28, 28)

    # Discriminator Network
    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(image_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, 1),
                nn.LeakyReLU(0.2)
            )
        
        def forward(self, img):
            img_flat = img.view(-1, image_dim)
            return self.model(img_flat)

    # Check if GPU is available and set it as a device (if not, choose CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize networks
    generator = Generator()
    discriminator = Discriminator()

    # Move models to GPU
    generator.to(device)
    discriminator.to(device)

    # Debug console output for printing the actual device
    print(next(generator.parameters()).device)
    print(next(discriminator.parameters()).device)

    # Initialize optimizers with Adam
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, beta2))
    g_optimizer = optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, beta2))

    # Learning rate scheduler
    g_scheduler = optim.lr_scheduler.StepLR(g_optimizer, step_size=30, gamma=0.9)
    d_scheduler = optim.lr_scheduler.StepLR(d_optimizer, step_size=30, gamma=0.9)

    # Loss function
    adversarial_loss = nn.BCEWithLogitsLoss()

    # Load MNIST data
    def load_mnist(path, kind='train'):
        """Load MNIST data from `path`"""
        labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte')
        images_path = os.path.join(path, f'{kind}-images-idx3-ubyte')

        with open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        with open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

        return images, labels

    # Path to your MNIST data
    data_path = r'C:/Users/uzivatel 1/OneDrive/Dokumenty/Coding Projects/Bachelor_Thesis/MNIST_data/raw/'

    # Load training data
    X_train, y_train = load_mnist(data_path, kind='train')

    # Convert to PyTorch tensors and normalize to [-1, 1]
    X_train = torch.FloatTensor(np.copy(X_train)).reshape(-1, 1, 28, 28) / 255.0
    X_train = (X_train - 0.5) / 0.5  # Normalize to [-1, 1]

    # Create dataset and dataloader
    train_dataset = TensorDataset(X_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Verify the dataset
    print(f"Dataset size: {len(train_dataset)}")
    print(f"First item shape: {train_dataset[0][0].shape}")
    print(f"Data range: [{X_train.min().item():.2f}, {X_train.max().item():.2f}]")

    # Function to add noise
    def add_noise(tensor, mean=0., std=0.1):
        return tensor + torch.randn(tensor.size(), device=tensor.device) * std + mean

    # Training loop
    #with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for epoch in range(num_epochs):
            total_d_loss = 0
            total_g_loss = 0
            batches = 0
            model_breaks = 0
            loss_threshold = 0.9

            for i, (real_images,) in enumerate(train_loader):
                batch_size = real_images.size(0)
            
                # Move real images to the GPU
                real_images = real_images.to(device)
            
                # Add noise to real images
                real_images = add_noise(real_images)

                # Ground truth labels
                valid = torch.ones(batch_size, 1).to(device)
                fake = torch.zeros(batch_size, 1).to(device)

                # Train Discriminator
                discriminator.zero_grad()
                real_loss = adversarial_loss(discriminator(real_images), valid)
            
                z = torch.randn(batch_size, latent_dim).to(device)
                fake_images = generator(z)
                fake_loss = adversarial_loss(discriminator(fake_images.detach()), fake)
            
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                d_optimizer.step()

                # Train Generator
                generator.zero_grad()
                g_loss = adversarial_loss(discriminator(fake_images), valid)
                g_loss.backward()
                g_optimizer.step()

                # Update running loss
                total_d_loss += d_loss.item()
                total_g_loss += g_loss.item()
                batches += 1

            # Calculate average loss for the epoch
            avg_d_loss = total_d_loss / batches
            avg_g_loss = total_g_loss / batches

            # Step the learning rate schedulers
            g_scheduler.step()
            d_scheduler.step()

            # Stop training if loss of one of the models is higher then loss_threshold
            if (avg_d_loss >= loss_threshold or avg_g_loss >= loss_threshold):
                model_breaks = 1
                print("Loss is too high to continue (model breaks)")
                break

            # Print losses occasionally
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], d_loss: {avg_d_loss:.4f}, g_loss: {avg_g_loss:.4f}")

            # Generate and save images occasionally
            if (epoch + 1) % 50 == 0:
                with torch.no_grad():
                    fake_images = generator(torch.randn(16, latent_dim).to(device)).reshape(-1, 28, 28)
                    img = fake_images.detach().cpu().numpy()
                    fig, axs = plt.subplots(4, 4, figsize=(10, 10))
                    for ax, img in zip(axs.flatten(), fake_images):
                        ax.imshow(img.detach().cpu().numpy(), cmap='gray')
                        ax.axis('off')
                plt.savefig(f'fake_images_epoch_{epoch+1}.png')
                plt.close()

        
        
   #print(prof.key_averages().table(sort_by="cuda_time_total"))

    # Save the trained models
    if (model_breaks == 0):
        print("Training finished!")
        torch.save(generator.state_dict(), 'generator.pth')
        torch.save(discriminator.state_dict(), 'discriminator.pth')
    
    print("Models of generator and disciminator will not be saved")