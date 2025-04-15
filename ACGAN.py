# Nazev souboru: ACGAN.py (opraveno pro torch-fidelity - ukladani uint8 a num_workers=0)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader # Dataset už není potřeba
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image
import pickle
from torch_fidelity import calculate_metrics
import torch.multiprocessing as mp
import tempfile
import shutil

# Uprav cestu k root adresáři, kam se stahuje MNIST (NE 'raw')
MNIST_DATASET_ROOT = "C:/Users/uzivatel 1/OneDrive/Dokumenty/Coding Projects/Bachelor_Thesis/MNIST_data"
# Adresář, kam se uloží PNG obrázky MNIST
MNIST_TRAIN_IMAGES_PATH = os.path.join(MNIST_DATASET_ROOT, "MNIST_train_images_png") # Doporučuji podsložku

def save_mnist_as_images(dataset_path_root, output_dir):
    """
    Načte MNIST trénovací dataset z dataset_path_root a uloží každý obrázek
    jako samostatný PNG soubor do output_dir.
    Spouští se pouze pokud output_dir neexistuje nebo je nekompletní.
    """
    num_mnist_train = 60000
    # Zkontroluj, zda adresář existuje a je kompletní
    if os.path.exists(output_dir):
        try:
            num_files = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
            if num_files == num_mnist_train:
                print(f"Adresář '{output_dir}' již existuje a zdá se být kompletní ({num_files} PNG). Přeskakuji generování.")
                return True # Adresář je připraven
            else:
                print(f"Adresář '{output_dir}' existuje, ale je nekompletní ({num_files}/{num_mnist_train} PNG). Mažu a generuji znovu.")
                shutil.rmtree(output_dir) # Smazat nekompletní adresář
        except OSError as e:
             print(f"Chyba při kontrole/mazání adresáře {output_dir}: {e}. Zkouším pokračovat...")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Generuji MNIST trénovací obrázky do adresáře '{output_dir}'...")

    # Načti MNIST dataset - bez normalizace, aby byl v rozsahu [0, 1] pro save_image
    transform_save = transforms.Compose([transforms.ToTensor()])
    try:
        # Použij root adresář, kde máš MNIST data (ne 'raw')
        mnist_dataset = datasets.MNIST(root=dataset_path_root, train=True, download=False, transform=transform_save)
        print(f"MNIST dataset nalezen v '{dataset_path_root}'")
    except RuntimeError as e:
        print(f"Nepodařilo se načíst MNIST z '{dataset_path_root}': {e}. Zkouším stáhnout...")
        try:
            mnist_dataset = datasets.MNIST(root=dataset_path_root, train=True, download=True, transform=transform_save)
            print(f"MNIST dataset úspěšně stažen do '{dataset_path_root}'")
        except Exception as download_e:
            print(f"FATAL: Nepodařilo se stáhnout ani načíst MNIST dataset z '{dataset_path_root}'. Chyba: {download_e}")
            return False # Chyba při získávání datasetu


    if len(mnist_dataset) != num_mnist_train:
         print(f"Varování: Očekávaný počet obrázků v MNIST trénovacím setu je {num_mnist_train}, nalezeno {len(mnist_dataset)}.")

    # Iteruj a ukládej obrázky
    count = 0
    try:
        for i in range(len(mnist_dataset)):
            image, label = mnist_dataset[i] # image je Tensor [1, 28, 28] v rozsahu [0, 1]
            # save_image očekává [C, H, W] nebo [B, C, H, W], náš image je [1, 28, 28] - to je v pořádku
            save_image(image, os.path.join(output_dir, f"mnist_train_{i:05d}.png"), normalize=False) # normalize=False, protože už je [0,1]
            count += 1
            if (i + 1) % 5000 == 0:
                print(f"  Uloženo {i + 1}/{len(mnist_dataset)} obrázků...")
        print(f"Úspěšně uloženo {count} MNIST trénovacích obrázků do '{output_dir}'.")
        return True # Vše OK
    except Exception as save_e:
        print(f"CHYBA při ukládání obrázku {count}: {save_e}")
        print("Ukládání přerušeno. Adresář může být nekompletní.")
        return False # Chyba při ukládání

# --- Definice modelů (MnistClassifier, Generator, Discriminator) ---
# (Kód pro všechny 3 třídy modelů zde - beze změny)
class MnistClassifier(nn.Module):
    def __init__(self):
        super(MnistClassifier, self).__init__()
        self.conv1=nn.Conv2d(1, 32, 5, 1, 2); self.pool1=nn.MaxPool2d(2, 2)
        self.conv2=nn.Conv2d(32, 64, 5, 1, 2); self.pool2=nn.MaxPool2d(2, 2)
        self.fc1=nn.Linear(64*7*7, 1024); self.fc2=nn.Linear(1024, 10)
    def forward(self, x):
        x=self.pool1(F.relu(self.conv1(x))); x=self.pool2(F.relu(self.conv2(x)))
        x=x.view(-1, 64*7*7); x=F.relu(self.fc1(x)); x=self.fc2(x); return x

class Generator(nn.Module):
    def __init__(self, z_dim=100, num_classes=10, img_channels=1):
        super(Generator, self).__init__(); self.z_dim=z_dim; self.num_classes=num_classes; self.img_channels=img_channels
        self.label_embedding = nn.Embedding(self.num_classes, self.num_classes); self.input = nn.Linear(self.z_dim + self.num_classes, 256 * 7 * 7)
        self.main = nn.Sequential(nn.BatchNorm2d(256), nn.Upsample(scale_factor=2), nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True), nn.Upsample(scale_factor=2), nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(64, self.img_channels, 3, 1, 1), nn.Tanh())
    def forward(self, z, labels):
        x=self.input(torch.cat([z, self.label_embedding(labels)], dim=1)); x=x.view(-1, 256, 7, 7); return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, num_classes=10, img_channels=1):
        super(Discriminator, self).__init__(); self.num_classes=num_classes; self.img_channels=img_channels
        self.main = nn.Sequential(nn.Conv2d(self.img_channels, 64, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.5), nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.5), nn.Conv2d(128, 256, 4, 1, 0), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.5))
        self.source_output = nn.Linear(256 * 4 * 4, 1); self.class_output = nn.Linear(256 * 4 * 4, self.num_classes); self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x=self.main(x); x=x.view(-1, 256*4*4); return self.sigmoid(self.source_output(x)), self.class_output(x)
# --- Konec definic modelů ---

# --- Funkce pro váhovou inicializaci ---
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1: nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1: nn.init.normal_(m.weight.data, 1.0, 0.02); nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1: nn.init.normal_(m.weight.data, 0.0, 0.02); nn.init.constant_(m.bias.data, 0)
# --- Konec funkce pro váhovou inicializaci ---

# --- Funkce pro evaluaci a ukládání vzorků ---
def save_samples(epoch, generator, fixed_noise, fixed_labels, output_dir, device):
    generator.eval()
    with torch.no_grad():
        fixed_noise = fixed_noise.to(device); fixed_labels = fixed_labels.to(device)
        gen_imgs = generator(fixed_noise, fixed_labels)
    fig, axs = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(10):
        row, col = i//5, i%5
        axs[row, col].imshow(gen_imgs[i].cpu().detach().squeeze(), cmap='gray'); axs[row, col].set_title(f"Třída: {fixed_labels[i].item()}"); axs[row, col].axis('off')
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, f"acgan_mnist_epoch_{epoch}_classes.png")); plt.close(fig)
    save_image(gen_imgs.data, os.path.join(output_dir, f"acgan_samples_epoch_{epoch}_grid.png"), nrow=5, normalize=True)
    print(f"Vzorky pro epochu {epoch} uloženy do '{output_dir}'")
    generator.train()

# --- Funkce pro evaluaci modelů - upravena pro ukládání uint8 a num_workers=0 ---
def evaluate_model(epoch, generator, classifier, device, z_dim, num_classes, batch_size,
                   num_samples_for_metrics, num_samples_for_accuracy,
                   mnist_images_path): # <<< Přidán parametr
    print(f"\n--- Evaluace Epochy {epoch} ---")
    generator.eval(); classifier.eval()
    fid_score, is_score, accuracy = None, None, None

    temp_dir = tempfile.mkdtemp()
    print(f"Generuji {num_samples_for_metrics} vzorků pro FID/IS do '{temp_dir}'...")
    img_counter = 0
    try:
        # ... (kód pro generování a ukládání obrázků do temp_dir - ten už funguje) ...
        # Stejný kód jako v předchozí funkční verzi:
        with torch.no_grad():
             num_batches = (num_samples_for_metrics + batch_size - 1) // batch_size
             for i in range(num_batches):
                current_batch_size = min(batch_size, num_samples_for_metrics - img_counter)
                if current_batch_size <= 0: break
                z = torch.randn(current_batch_size, z_dim, device=device)
                labels_fid = torch.randint(0, num_classes, (current_batch_size,), device=device)
                imgs = generator(z, labels_fid) # imgs je v rozsahu [-1, 1] FloatTensor
                for img_tensor in imgs.detach():
                    if img_counter >= num_samples_for_metrics: break
                    save_image(
                        img_tensor,
                        os.path.join(temp_dir, f"img_{img_counter:05d}.png"),
                        normalize=True # save_image převede [-1, 1] na [0, 255] uint8
                    )
                    img_counter += 1
                if img_counter >= num_samples_for_metrics: break


        # --- Volání calculate_metrics s cestou k MNIST PNG ---
        if img_counter >= num_samples_for_metrics:
            print(f"Obrázky uloženy ({img_counter}), počítám FID pomocí torch-fidelity...")
            print(f"  Input 1 (generované): {temp_dir}")
            print(f"  Input 2 (referenční): {mnist_images_path}") # Vypíše cestu
            try:
                metrics_dict = calculate_metrics(
                    input1=temp_dir,          # Adresář s generovanými PNG
                    input2=mnist_images_path, # <<< Adresář s referenčními MNIST PNG
                    cuda=torch.cuda.is_available(),
                    isc=True,                # Stále vypnuto pro test
                    fid=True,
                    batch_size=batch_size,    # Můžeš zkusit snížit, pokud by byla chyba paměti
                    num_workers=0,            # Stále 0 kvůli Windows
                    verbose=True              # Přidáno pro více info od torch-fidelity
                )
                
                # Získání výsledků z dictionary
                fid_score = metrics_dict.get('frechet_inception_distance', float('nan')) # Bezpečnější get i pro FID pro jistotu
                is_mean = metrics_dict.get('inception_score_mean', float('nan'))
                is_std = metrics_dict.get('inception_score_std', float('nan'))

                # Výpis výsledků
                if not np.isnan(fid_score):
                    print(f"FID: {fid_score:.4f}")
                else:
                    print("FID: Nebylo vypočítáno.")

                if not np.isnan(is_mean):
                    print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}") # Vypíšeme i směrodatnou odchylku
                else:
                    print("Inception Score: Nebylo vypočítáno.")

                # Uložení průměrné hodnoty IS pro logování/návrat z funkce
                is_score = is_mean

            except Exception as e:
                print(f"CHYBA při výpočtu metrik: {e}")
                import traceback
                traceback.print_exc() # Podrobný traceback
                print("Výpočet metrik přeskočen.")
                fid_score, is_score = float('nan'), float('nan')
        else:
            print(f"Nedostatek vygenerovaných vzorků ({img_counter}/{num_samples_for_metrics}) pro metriky.")
            fid_score, is_score = float('nan'), float('nan')

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir); print(f"Dočasný adresář '{temp_dir}' smazán.")
    
    # --- Výpočet přesnosti klasifikace (beze změny) ---
    print(f"Generuji {num_samples_for_accuracy} vzorků pro test klasifikace...")
    all_preds, all_targets = [], []
    num_samples_per_class = num_samples_for_accuracy // num_classes
    with torch.no_grad():
        for digit in range(num_classes):
            z = torch.randn(num_samples_per_class, z_dim, device=device)
            target_labels = torch.LongTensor([digit] * num_samples_per_class).to(device)
            gen_imgs = generator(z, target_labels)
            outputs = classifier(gen_imgs); _, predicted = torch.max(outputs.data, 1)
            all_preds.append(predicted.cpu()); all_targets.append(target_labels.cpu())
    if all_preds:
        all_preds = torch.cat(all_preds); all_targets = torch.cat(all_targets)
        correct = (all_preds == all_targets).sum().item()
        accuracy = 100. * correct / len(all_targets) if len(all_targets) > 0 else 0.0
        print(f"Přesnost klasifikace generovaných obrázků: {accuracy:.2f}% ({correct}/{len(all_targets)})")
    else:
        accuracy = float('nan'); print("Přesnost klasifikace nebyla vypočtena (žádné vzorky).")

    # ... (výpočet přesnosti klasifikace) ...

    generator.train()
    print(f"--- Konec Evaluace Epochy {epoch} ---")
    return fid_score, is_score, accuracy
# --- Konec funkcí pro evaluaci ---

# --- Hlavní část skriptu ---
if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True); print("Metoda startu multiprocesingu nastavena na 'spawn'.")
    except RuntimeError: print("Metoda startu multiprocesingu byla již nastavena.")

    # --- Krok 1: Zajisti existenci MNIST obrázků ve formátu PNG ---
    print("-" * 40)
    print("Kontrola/Generování MNIST PNG obrázků pro FID...")
    # Zavolej funkci pro uložení. Pokud selže, ukonči skript.
    if not save_mnist_as_images(MNIST_DATASET_ROOT, MNIST_TRAIN_IMAGES_PATH):
         print("Chyba při přípravě MNIST PNG obrázků. Výpočet FID nebude možný. Ukončuji.")
         exit()
    print("MNIST PNG obrázky jsou připraveny.")
    print("-" * 40)

    # --- Zbytek kódu uvnitř if __name__ == '__main__': ---
    # (Nastavení, načítání dat, inicializace modelů, loss, optimizéry atd. - beze změny)
    torch.manual_seed(42); np.random.seed(42)
    MNIST_PATH = "C:/Users/uzivatel 1/OneDrive/Dokumenty/Coding Projects/Bachelor_Thesis/MNIST_data/raw"
    batch_size = 64; z_dim = 100; num_classes = 10; img_channels = 1
    learning_rate = 0.0002; betas = (0.5, 0.999); num_epochs = 50
    LOG_INTERVAL = 100; EVAL_INTERVAL = 5; NUM_SAMPLES_FOR_METRICS = 5000
    NUM_SAMPLES_FOR_ACCURACY = 1000; CLASSIFIER_PATH = "mnist_classifier.pth"
    LOG_FILE = "acgan_training_logs.pkl"; OUTPUT_DIR_IMAGES = "acgan_images_output"; OUTPUT_DIR_MODELS = "acgan_models_output"
    MNIST_DOWNLOAD_PATH = os.path.join(MNIST_DATASET_ROOT)
    os.makedirs(OUTPUT_DIR_IMAGES, exist_ok=True); os.makedirs(OUTPUT_DIR_MODELS, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Použité zařízení: {device}")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    try:
        # Pro trénink použij transformaci s normalizací (-1, 1)
        transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = datasets.MNIST(root=MNIST_DOWNLOAD_PATH, train=True, transform=transform_train, download=False) # download=False, už bychom měli mít
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) # num_workers=0 zde může být kvůli Windows
        print(f"Tréninkový dataset úspěšně načten: {len(train_dataset)} obrázků")
    except Exception as e:
        print(f"Nepodařilo se načíst tréninkový dataset: {e}\nZkusím stáhnout dataset...")
        # Pokud stahování selhalo i v save_mnist_as_images, zde to asi také selže
        train_dataset = datasets.MNIST(root=MNIST_DOWNLOAD_PATH, train=True, transform=transform_train, download=True)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        print(f"Tréninkový dataset úspěšně stažen: {len(train_dataset)} obrázků")
    generator=Generator(z_dim=z_dim,num_classes=num_classes,img_channels=img_channels).to(device); discriminator=Discriminator(num_classes=num_classes,img_channels=img_channels).to(device)
    generator.apply(weights_init); discriminator.apply(weights_init)
    adversarial_loss=nn.BCELoss(); auxiliary_loss=nn.CrossEntropyLoss()
    optimizer_G=optim.Adam(generator.parameters(), lr=learning_rate, betas=betas); optimizer_D=optim.Adam(discriminator.parameters(), lr=learning_rate, betas=betas)
    classifier=MnistClassifier().to(device)
    try:
        classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device)); classifier.eval(); print(f"Externí klasifikátor '{CLASSIFIER_PATH}' úspěšně načten.")
    except FileNotFoundError: print(f"CHYBA: Soubor klasifikátoru '{CLASSIFIER_PATH}' nebyl nalezen.\nProsím, nejprve spusťte 'train_classifier.py'."); exit()
    except Exception as e: print(f"CHYBA: Nepodařilo se načíst klasifikátor '{CLASSIFIER_PATH}': {e}"); exit()
    fixed_noise=torch.randn(10, z_dim); fixed_labels=torch.LongTensor(np.arange(10))
    training_logs={'iterations':[], 'epochs':[], 'g_loss':[], 'd_loss':[], 'd_real_loss':[], 'd_fake_loss':[], 'd_aux_loss_real':[], 'd_aux_loss_fake':[], 'g_aux_loss':[], 'eval_epochs':[], 'fid_scores':[], 'is_scores':[], 'classification_accuracy':[]}

    best_fid = float('inf') # Nejlepší FID inicializujeme na nekonečno
    best_epoch = -1
    os.makedirs(OUTPUT_DIR_MODELS, exist_ok=True)

    # --- Tréninkový cyklus (beze změny) ---
    print("Zahajuji trénink ACGAN...")
    total_iterations = 0
    for epoch in range(num_epochs):
        epoch_g_loss = 0.0; epoch_d_loss = 0.0; num_batches = len(train_loader)
        discriminator.train(); generator.train()
        for i, (real_imgs, labels) in enumerate(train_loader):
            batch_size_current=real_imgs.size(0); real_labels_smooth=torch.full((batch_size_current,1), 0.9, device=device); fake_labels=torch.zeros(batch_size_current,1).to(device)
            real_imgs=real_imgs.to(device); labels=labels.to(device)
            optimizer_D.zero_grad()
            validity_real, class_logits_real = discriminator(real_imgs); d_real_adv_loss=adversarial_loss(validity_real, real_labels_smooth); d_real_aux_loss=auxiliary_loss(class_logits_real, labels)
            z=torch.randn(batch_size_current, z_dim).to(device); gen_labels=torch.randint(0,num_classes,(batch_size_current,)).to(device); gen_imgs=generator(z, gen_labels).detach()
            validity_fake, class_logits_fake = discriminator(gen_imgs); d_fake_adv_loss=adversarial_loss(validity_fake, fake_labels); d_fake_aux_loss=auxiliary_loss(class_logits_fake, gen_labels)
            d_adv_loss=(d_real_adv_loss+d_fake_adv_loss)/2; d_aux_loss=(d_real_aux_loss+d_fake_aux_loss)/2; d_loss=d_adv_loss+d_aux_loss
            if torch.isnan(d_loss): print(f"ERROR: NaN D loss v epoch {epoch}, iter {i}. Ukončuji."); exit()
            d_loss.backward(); optimizer_D.step()
            optimizer_G.zero_grad()
            z_g=torch.randn(batch_size_current, z_dim).to(device); gen_labels_g=torch.randint(0,num_classes,(batch_size_current,)).to(device)
            gen_imgs_g=generator(z_g, gen_labels_g); validity_g, class_logits_g = discriminator(gen_imgs_g)
            g_adv_loss=adversarial_loss(validity_g, torch.ones(batch_size_current,1, device=device)); g_aux_loss=auxiliary_loss(class_logits_g, gen_labels_g); g_loss=g_adv_loss+g_aux_loss
            if torch.isnan(g_loss): print(f"ERROR: NaN G loss v epoch {epoch}, iter {i}. Ukončuji."); exit()
            g_loss.backward(); optimizer_G.step()
            epoch_g_loss += g_loss.item(); epoch_d_loss += d_loss.item()
            current_iter = epoch * num_batches + i
            if current_iter % LOG_INTERVAL == 0 and not (torch.isnan(d_loss) or torch.isnan(g_loss)):
                training_logs['iterations'].append(current_iter); training_logs['g_loss'].append(g_loss.item()); training_logs['d_loss'].append(d_loss.item())
                training_logs['d_real_loss'].append(d_real_adv_loss.item()); training_logs['d_fake_loss'].append(d_fake_adv_loss.item())
                training_logs['d_aux_loss_real'].append(d_real_aux_loss.item()); training_logs['d_aux_loss_fake'].append(d_fake_aux_loss.item()); training_logs['g_aux_loss'].append(g_aux_loss.item())
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{num_batches}] [Iter {current_iter}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
            total_iterations += 1
        avg_g_loss = epoch_g_loss/num_batches if num_batches>0 else 0
        avg_d_loss = epoch_d_loss/num_batches if num_batches>0 else 0
        print(f"\n=== End of Epoch {epoch} ===\nAvg G Loss: {avg_g_loss:.4f}, Avg D Loss: {avg_d_loss:.4f}")
        training_logs['epochs'].append(epoch)
        save_samples(epoch, generator, fixed_noise, fixed_labels, OUTPUT_DIR_IMAGES, device)

        # --- Evaluace a Checkpointing ---
        if epoch % EVAL_INTERVAL == 0 or epoch == num_epochs - 1:
            # Předáváme cestu k PNG obrázkům MNIST do evaluace
            fid, isc, acc = evaluate_model(epoch, generator, classifier, device, z_dim, num_classes, batch_size,
                                            NUM_SAMPLES_FOR_METRICS, NUM_SAMPLES_FOR_ACCURACY,
                                            MNIST_TRAIN_IMAGES_PATH) # <<< Předání cesty
            training_logs['eval_epochs'].append(epoch)
            training_logs['fid_scores'].append(fid)
            training_logs['is_scores'].append(isc) # Přidej IS, až ho zapneš
            training_logs['classification_accuracy'].append(acc)

            # --- Logika pro ukládání nejlepšího modelu ---
            if not np.isnan(fid) and fid < best_fid:
                best_fid = fid
                best_epoch = epoch
                # Uložení modelu s lepším FID
                best_generator_save_path = os.path.join(OUTPUT_DIR_MODELS, f'acgan_generator_best.pth') # Jednoduchý název, přepíše se
                best_discriminator_save_path = os.path.join(OUTPUT_DIR_MODELS, f'acgan_discriminator_best.pth')
                # Nebo ukládej s epochou a FID v názvu, pokud chceš historii:
                # best_generator_save_path = os.path.join(OUTPUT_DIR_MODELS, f'acgan_generator_ep{epoch}_fid{fid:.2f}.pth')
                # best_discriminator_save_path = os.path.join(OUTPUT_DIR_MODELS, f'acgan_discriminator_ep{epoch}_fid{fid:.2f}.pth')

                try:
                    torch.save(generator.state_dict(), best_generator_save_path)
                    torch.save(discriminator.state_dict(), best_discriminator_save_path)
                    print(f"*** Nové nejlepší FID: {best_fid:.4f} v epoše {best_epoch}. Modely uloženy jako '..._best.pth'. ***")
                except Exception as e:
                    print(f"CHYBA při ukládání nejlepšího modelu: {e}")
            # --- Konec logiky pro ukládání ---

        print("=========================\n")

    # --- Ukládání a finální generování (beze změny) ---
    print("Trénink dokončen!")
    print(f"Nejlepší FID dosaženo v epoše {best_epoch} s hodnotou: {best_fid:.4f}")
    try:
        with open(LOG_FILE, 'wb') as f: pickle.dump(training_logs, f); print(f"Tréninkové logy uloženy do: {LOG_FILE}")
    except Exception as e: print(f"CHYBA při ukládání logů: {e}")
    generator_save_path = os.path.join(OUTPUT_DIR_MODELS, 'acgan_generator_final.pth'); discriminator_save_path = os.path.join(OUTPUT_DIR_MODELS, 'acgan_discriminator_final.pth')
    torch.save(generator.state_dict(), generator_save_path); torch.save(discriminator.state_dict(), discriminator_save_path); print(f"Finální modely uloženy do: '{OUTPUT_DIR_MODELS}'")
    print("Generuji finální vzorky pro každou třídu..."); generator.eval()
    with torch.no_grad():
        rows, cols = 10, 10; samples_per_class = cols; all_final_samples = []
        for digit in range(rows):
            z=torch.randn(samples_per_class, z_dim, device=device); labels=torch.LongTensor([digit]*samples_per_class).to(device)
            gen_imgs=generator(z, labels); all_final_samples.append(gen_imgs.cpu())
            save_image(gen_imgs.data, os.path.join(OUTPUT_DIR_IMAGES, f"acgan_final_digit_{digit}.png"), nrow=cols, normalize=True)
        if all_final_samples:
             all_final_samples_tensor = torch.cat(all_final_samples)
             save_image(all_final_samples_tensor.data, os.path.join(OUTPUT_DIR_IMAGES, "acgan_final_all_digits_grid.png"), nrow=cols, normalize=True)
        print(f"Finální vzorky uloženy do '{OUTPUT_DIR_IMAGES}'.")