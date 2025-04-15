# Nazev souboru: analyze_results.py
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Konfigurace ---
LOG_FILE = "acgan_training_logs.pkl" # Cesta k uloženým logům
OUTPUT_DIR_PLOTS = "analysis_plots" # Adresář pro grafy
os.makedirs(OUTPUT_DIR_PLOTS, exist_ok=True)

# --- Načtení logů ---
try:
    with open(LOG_FILE, 'rb') as f:
        logs = pickle.load(f)
    print(f"Logy úspěšně načteny z '{LOG_FILE}'")
except FileNotFoundError:
    print(f"CHYBA: Soubor s logy '{LOG_FILE}' nebyl nalezen.")
    print("Ujistěte se, že jste nejprve spustili upravený ACGAN.py skript.")
    exit()
except Exception as e:
    print(f"CHYBA při načítání logů: {e}")
    exit()

# --- Zpracování dat pro grafy ---
iterations = logs.get('iterations', [])
epochs_loss = np.array(iterations) / (max(iterations)/ (logs.get('epochs',[-1])[-1]+1) if iterations and logs.get('epochs') else 1) # Odhad epochy pro loss
g_loss = logs.get('g_loss', [])
d_loss = logs.get('d_loss', [])
d_real_loss = logs.get('d_real_loss', [])
d_fake_loss = logs.get('d_fake_loss', [])
d_aux_loss_real = logs.get('d_aux_loss_real', [])
d_aux_loss_fake = logs.get('d_aux_loss_fake', [])
g_aux_loss = logs.get('g_aux_loss', [])

eval_epochs = logs.get('eval_epochs', [])
fid_scores = logs.get('fid_scores', [])
is_scores = logs.get('is_scores', [])
accuracy = logs.get('classification_accuracy', [])

# Odstranění NaN hodnot pro kreslení (matplotlib si s nimi neporadí dobře v linkách)
valid_fid_indices = ~np.isnan(fid_scores)
valid_is_indices = ~np.isnan(is_scores)
valid_acc_indices = ~np.isnan(accuracy)

eval_epochs_fid = np.array(eval_epochs)[valid_fid_indices]
fid_scores_valid = np.array(fid_scores)[valid_fid_indices]

eval_epochs_is = np.array(eval_epochs)[valid_is_indices]
is_scores_valid = np.array(is_scores)[valid_is_indices]

eval_epochs_acc = np.array(eval_epochs)[valid_acc_indices]
accuracy_valid = np.array(accuracy)[valid_acc_indices]


# --- Generování Grafů ---
print("Generuji grafy...")

# 1. Graf G a D Loss
plt.figure(figsize=(12, 6))
plt.plot(epochs_loss, g_loss, label='Generator Loss')
plt.plot(epochs_loss, d_loss, label='Discriminator Loss')
plt.xlabel("Přibližná Epocha")
plt.ylabel("Loss")
plt.title("Vývoj Loss Funkcí Generátoru a Diskriminátoru")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, "loss_G_vs_D.png"))
plt.close()

# 2. Graf Adversariálních Loss Diskriminátoru
plt.figure(figsize=(12, 6))
plt.plot(epochs_loss, d_real_loss, label='D Real Adv Loss')
plt.plot(epochs_loss, d_fake_loss, label='D Fake Adv Loss')
plt.xlabel("Přibližná Epocha")
plt.ylabel("Adversarial Loss")
plt.title("Vývoj Adversariálních Loss Funkcí Diskriminátoru")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, "loss_D_adversarial.png"))
plt.close()

# 3. Graf Auxiliary Loss
plt.figure(figsize=(12, 6))
plt.plot(epochs_loss, d_aux_loss_real, label='D Real Aux Loss')
plt.plot(epochs_loss, d_aux_loss_fake, label='D Fake Aux Loss')
plt.plot(epochs_loss, g_aux_loss, label='G Aux Loss')
plt.xlabel("Přibližná Epocha")
plt.ylabel("Auxiliary Loss")
plt.title("Vývoj Auxiliary (klasifikačních) Loss Funkcí")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, "loss_Auxiliary.png"))
plt.close()

# 4. Graf FID skóre
if len(eval_epochs_fid) > 0:
    plt.figure(figsize=(10, 5))
    plt.plot(eval_epochs_fid, fid_scores_valid, marker='o', label='FID Score')
    plt.xlabel("Epocha")
    plt.ylabel("FID Score (nižší je lepší)")
    plt.title("Vývoj FID Skóre Během Tréninku")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, "metric_FID.png"))
    plt.close()
else:
    print("Žádná validní data pro FID graf.")


# 5. Graf Inception Score (IS)
if len(eval_epochs_is) > 0:
    plt.figure(figsize=(10, 5))
    plt.plot(eval_epochs_is, is_scores_valid, marker='o', label='Inception Score')
    plt.xlabel("Epocha")
    plt.ylabel("Inception Score (vyšší je lepší)")
    plt.title("Vývoj Inception Score Během Tréninku")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, "metric_IS.png"))
    plt.close()
else:
    print("Žádná validní data pro IS graf.")

# 6. Graf Přesnosti Klasifikace
if len(eval_epochs_acc) > 0:
    plt.figure(figsize=(10, 5))
    plt.plot(eval_epochs_acc, accuracy_valid, marker='o', label='Classification Accuracy')
    plt.xlabel("Epocha")
    plt.ylabel("Přesnost (%)")
    plt.title("Přesnost Klasifikace Generovaných Obrázků Externím Klasifikátorem")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 100) # Osa Y od 0 do 100
    plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, "metric_Accuracy.png"))
    plt.close()
else:
    print("Žádná validní data pro graf přesnosti klasifikace.")


print(f"Grafy uloženy do adresáře '{OUTPUT_DIR_PLOTS}'.")

# --- Výpis Finálních Metrik (pro snadné kopírování do práce) ---
print("\n--- Finální Metriky (poslední měřená epocha) ---")
last_eval_epoch = eval_epochs[-1] if eval_epochs else "N/A"
last_fid = fid_scores[-1] if fid_scores else "N/A"
last_is = is_scores[-1] if is_scores else "N/A"
last_acc = accuracy[-1] if accuracy else "N/A"

print(f"Poslední evaluovaná epocha: {last_eval_epoch}")
print(f"Finální FID: {last_fid if isinstance(last_fid, str) else f'{last_fid:.4f}'}")
print(f"Finální Inception Score: {last_is if isinstance(last_is, str) else f'{last_is:.4f}'}")
print(f"Finální Přesnost Klasifikace: {last_acc if isinstance(last_acc, str) else f'{last_acc:.2f}%'}")
print("-----------------------------------------------")

# Můžete přidat i nejlepší hodnoty, pokud je to relevantní
best_fid_epoch = eval_epochs_fid[np.argmin(fid_scores_valid)] if len(eval_epochs_fid) > 0 else "N/A"
best_fid = np.min(fid_scores_valid) if len(eval_epochs_fid) > 0 else "N/A"
best_is_epoch = eval_epochs_is[np.argmax(is_scores_valid)] if len(eval_epochs_is) > 0 else "N/A"
best_is = np.max(is_scores_valid) if len(eval_epochs_is) > 0 else "N/A"
best_acc_epoch = eval_epochs_acc[np.argmax(accuracy_valid)] if len(eval_epochs_acc) > 0 else "N/A"
best_acc = np.max(accuracy_valid) if len(eval_epochs_acc) > 0 else "N/A"

print("\n--- Nejlepší Dosažené Metriky ---")
print(f"Nejlepší FID: {best_fid:.4f} (v epoše {best_fid_epoch})" if best_fid != "N/A" else "Nejlepší FID: N/A")
print(f"Nejlepší IS: {best_is:.4f} (v epoše {best_is_epoch})" if best_is != "N/A" else "Nejlepší IS: N/A")
print(f"Nejlepší Přesnost: {best_acc:.2f}% (v epoše {best_acc_epoch})" if best_acc != "N/A" else "Nejlepší Přesnost: N/A")
print("---------------------------------")