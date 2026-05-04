"""
Question 5 – Neural Networks: Regularisation, Transfer Learning, and Interpretability
CIFAR-10 Dataset | PyTorch Implementation

HOW TO USE IN GOOGLE COLAB:
  1. Upload your cifar-10-python.tar.gz file to Colab using the file panel
     (click the folder icon on the left sidebar, then the upload button).
  2. Run this script. It will automatically extract the dataset and proceed.
  3. All output plots are saved to the q5_outputs/ folder.

  If your file has a different name (e.g. cifar-10.tar.gz), update TAR_PATH below.
"""

# ─────────────────────────────────────────────────────────────
# STEP 0 — Extract CIFAR-10 from the uploaded tar.gz
# ─────────────────────────────────────────────────────────────
import os
import tarfile

TAR_PATH  = "cifar-10-python.tar.gz"          # ← update if your filename differs
DATA_ROOT = "./data"
EXTRACTED = os.path.join(DATA_ROOT, "cifar-10-batches-py")

if os.path.exists(EXTRACTED):
    print("✓ CIFAR-10 already extracted, skipping.")
else:
    if not os.path.exists(TAR_PATH):
        raise FileNotFoundError(
            f"\n[ERROR] Could not find '{TAR_PATH}'.\n"
            "Please upload cifar-10-python.tar.gz to the Colab file panel\n"
            "(folder icon on the left sidebar → upload button).\n"
            "If your file has a different name, change TAR_PATH at the top of this script."
        )
    print(f"Extracting {TAR_PATH} → {DATA_ROOT}/ ...")
    os.makedirs(DATA_ROOT, exist_ok=True)
    with tarfile.open(TAR_PATH, "r:gz") as tar:
        tar.extractall(DATA_ROOT)
    print("✓ Extraction complete.")

# ─────────────────────────────────────────────────────────────
# STEP 1 — Imports
# ─────────────────────────────────────────────────────────────
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import optuna
import warnings
import cv2
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# STEP 2 — Reproducibility & Global Config
# ─────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

CLASSES     = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
NUM_CLASSES = 10
SAVE_DIR    = "q5_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# STEP 3 — Transforms & Data Loaders
# ─────────────────────────────────────────────────────────────
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# For ResNet18 — ImageNet preprocessing at 224×224
pretrained_transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
pretrained_transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def get_loaders(transform_train, transform_test, batch_size=128, val_ratio=0.1):
    """Build train / val / test DataLoaders from the locally extracted dataset."""
    full_train = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True,  download=False, transform=transform_train)
    test_set   = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=False, transform=transform_test)

    val_size   = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────
# STEP 4 — Model Definitions
# ─────────────────────────────────────────────────────────────

class DeepMLP(nn.Module):
    """Deep MLP: ≥4 hidden layers (512 neurons each), BatchNorm, Dropout."""
    def __init__(self, dropout_rate=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(512, 512),          nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(512, 512),          nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(512, 512),          nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(512, 256),          nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_rate / 2),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    """CNN: 3 conv-blocks (64→128→256), BatchNorm, Dropout2d, MaxPool."""
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),   nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),  nn.BatchNorm2d(64),  nn.ReLU(),
            nn.MaxPool2d(2, 2),               nn.Dropout2d(dropout_rate / 2),
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),  nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),                nn.Dropout2d(dropout_rate / 2),
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2),                nn.Dropout2d(dropout_rate),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(512, NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_pretrained_resnet(fine_tune_layers=2):
    """ResNet18 pre-trained on ImageNet; only last `fine_tune_layers` groups are unfrozen."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():                              # freeze all
        param.requires_grad = False
    for child in list(model.children())[-fine_tune_layers:]:     # unfreeze last N
        for param in child.parameters():
            param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)      # replace head
    return model


# ─────────────────────────────────────────────────────────────
# STEP 5 — Training & Evaluation Utilities
# ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scaler=None):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                out  = model(X)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out  = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * X.size(0)
        correct    += (out.argmax(1) == y).sum().item()
        total      += X.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for X, y in loader:
        X, y  = X.to(DEVICE), y.to(DEVICE)
        out   = model(X)
        loss  = criterion(out, y)
        total_loss += loss.item() * X.size(0)
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total   += X.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def top5_error(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            top5 = model(X).topk(5, dim=1).indices
            correct += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
            total   += X.size(0)
    return 1 - correct / total


def full_train(model, train_loader, val_loader, epochs=50, lr=1e-3,
               weight_decay=1e-4, patience=10, tag="model"):
    """Train with AdamW + cosine LR schedule + early stopping. Returns best model."""
    model     = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = torch.cuda.amp.GradScaler() if DEVICE.type == "cuda" else None

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss, best_state, wait = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        tl, ta       = train_epoch(model, train_loader, optimizer, criterion, scaler)
        vl, va, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)

        if vl < best_val_loss:
            best_val_loss = vl
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            wait          = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            print(f"  [{tag}] Epoch {epoch:3d} | "
                  f"Train Loss {tl:.4f} Acc {ta:.4f} | "
                  f"Val Loss {vl:.4f} Acc {va:.4f}")

    model.load_state_dict(best_state)
    return model, history


# ─────────────────────────────────────────────────────────────
# STEP 6 — Hyperparameter Optimisation (Optuna)
# ─────────────────────────────────────────────────────────────

def optuna_objective_mlp(trial):
    lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size   = trial.suggest_categorical("batch_size", [64, 128, 256])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

    tl, vl, _ = get_loaders(train_transform, test_transform, batch_size=batch_size)
    model      = DeepMLP(dropout_rate=dropout_rate).to(DEVICE)
    criterion  = nn.CrossEntropyLoss()
    optimizer  = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    for epoch in range(15):
        train_epoch(model, tl, optimizer, criterion)
        val_loss, _, _, _ = evaluate(model, vl, criterion)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        best_val = min(best_val, val_loss)
    return best_val


def run_optuna_mlp(n_trials=15):
    print("\n=== Optuna hyperparameter search for MLP ===")
    study = optuna.create_study(direction="minimize",
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(optuna_objective_mlp, n_trials=n_trials, show_progress_bar=True)
    print(f"  Best params: {study.best_params}")
    return study.best_params


# ─────────────────────────────────────────────────────────────
# STEP 7 — Plotting Utilities
# ─────────────────────────────────────────────────────────────

def plot_history(histories: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for tag, h in histories.items():
        axes[0].plot(h["train_loss"], label=f"{tag} train")
        axes[0].plot(h["val_loss"],   label=f"{tag} val", linestyle="--")
        axes[1].plot(h["train_acc"],  label=f"{tag} train")
        axes[1].plot(h["val_acc"],    label=f"{tag} val",  linestyle="--")
    for ax, ylabel in zip(axes, ["Loss", "Accuracy"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)
    axes[0].set_title("Loss"); axes[1].set_title("Accuracy")
    fig.suptitle("Training Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "training_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(labels, preds, tag="model"):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix – {tag}")
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, f"cm_{tag}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")


def print_metrics(labels, preds, model, test_loader, tag):
    acc   = (labels == preds).mean()
    macro = f1_score(labels, preds, average="macro")
    top5e = top5_error(model, test_loader)
    print(f"\n{'='*55}\n  Model: {tag}")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Macro-F1    : {macro:.4f}")
    print(f"  Top-5 Error : {top5e:.4f}")
    print(classification_report(labels, preds, target_names=CLASSES))
    return {"accuracy": acc, "macro_f1": macro, "top5_error": top5e}


# ─────────────────────────────────────────────────────────────
# STEP 8 — Grad-CAM
# ─────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, x, class_idx=None):
        self.model.eval()
        x   = x.to(DEVICE).unsqueeze(0)
        out = self.model(x)
        if class_idx is None:
            class_idx = out.argmax(1).item()
        self.model.zero_grad()
        out[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1).squeeze()
        cam     = F.relu(cam)
        cam     = cam - cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam.cpu().numpy(), class_idx


def visualise_gradcam(model, target_layer, test_loader, tag="CNN", n=10):
    """Collect n misclassified examples and plot Grad-CAM overlays."""
    
    gradcam = GradCAM(model, target_layer)
    model.eval()

    misclassified = []
    for X, y in test_loader:
        with torch.no_grad():
            preds = model(X.to(DEVICE)).argmax(1).cpu()
        wrong = (preds != y).nonzero(as_tuple=True)[0]
        for idx in wrong:
            if len(misclassified) >= n:
                break
            misclassified.append((X[idx], y[idx].item(), preds[idx].item()))
        if len(misclassified) >= n:
            break

    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    inv_mean = torch.tensor(MEAN).view(3, 1, 1)
    inv_std  = torch.tensor(STD).view(3, 1, 1)

    for i, (img, true_cls, pred_cls) in enumerate(misclassified):
        cam, _ = gradcam.generate(img, pred_cls)
        img_np      = (img * inv_std + inv_mean).clamp(0, 1).permute(1, 2, 0).numpy()
        cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
        heatmap     = plt.cm.jet(cam_resized)[..., :3]
        overlay     = 0.5 * img_np + 0.5 * heatmap

        axes[2 * i].imshow(img_np)
        axes[2 * i].set_title(
            f"True: {CLASSES[true_cls]}\nPred: {CLASSES[pred_cls]}", fontsize=7)
        axes[2 * i].axis("off")
        axes[2 * i + 1].imshow(overlay)
        axes[2 * i + 1].set_title("Grad-CAM", fontsize=7)
        axes[2 * i + 1].axis("off")

    for j in range(2 * len(misclassified), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Grad-CAM – {tag} Misclassifications", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, f"gradcam_{tag}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────
# STEP 9 — SHAP for MLP
# ─────────────────────────────────────────────────────────────

def explain_mlp_shap(model, test_loader, n_samples=50):
    try:
        import shap
    except ImportError:
        print("  SHAP not installed — run:  pip install shap")
        return

    model.eval()
    X_bg, _   = next(iter(test_loader))
    X_test, _ = next(iter(test_loader))
    X_bg   = X_bg[:100].cpu().numpy().reshape(100, -1)
    X_test = X_test[:n_samples].cpu().numpy().reshape(n_samples, -1)

    def predict(x_flat):
        t = torch.tensor(x_flat, dtype=torch.float32).reshape(-1, 3, 32, 32).to(DEVICE)
        with torch.no_grad():
            return model(t).cpu().numpy()

    explainer   = shap.KernelExplainer(predict, X_bg)
    shap_values = explainer.shap_values(X_test, nsamples=50)

    shap.summary_plot(shap_values[0], X_test, show=False, max_display=20)
    path = os.path.join(SAVE_DIR, "shap_mlp.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────
# STEP 10 — Adversarial Robustness (FGSM + PGD)
# ─────────────────────────────────────────────────────────────

def fgsm_attack(model, X, y, epsilon=0.03):
    X_adv = X.clone().detach().requires_grad_(True).to(DEVICE)
    loss  = nn.CrossEntropyLoss()(model(X_adv), y.to(DEVICE))
    loss.backward()
    return (X_adv + epsilon * X_adv.grad.sign()).detach()


def pgd_attack(model, X, y, epsilon=0.03, alpha=0.01, steps=10):
    X_adv = X.clone().detach().to(DEVICE)
    y     = y.to(DEVICE)
    for _ in range(steps):
        X_adv.requires_grad_(True)
        loss = nn.CrossEntropyLoss()(model(X_adv), y)
        loss.backward()
        with torch.no_grad():
            X_adv = X_adv + alpha * X_adv.grad.sign()
            delta = torch.clamp(X_adv - X.to(DEVICE), -epsilon, epsilon)
            X_adv = (X.to(DEVICE) + delta).detach()
    return X_adv


@torch.no_grad()
def acc_on_batch(model, X, y):
    return (model(X.to(DEVICE)).argmax(1) == y.to(DEVICE)).float().mean().item()


def robustness_eval(models_dict, test_loader, epsilon=0.03, n_batches=10):
    print("\n=== Adversarial Robustness Evaluation ===")
    results = {tag: {"clean": [], "fgsm": [], "pgd": []} for tag in models_dict}

    for i, (X, y) in enumerate(test_loader):
        if i >= n_batches:
            break
        X, y = X.to(DEVICE), y.to(DEVICE)
        for tag, model in models_dict.items():
            model.eval()
            results[tag]["clean"].append(acc_on_batch(model, X, y))
            results[tag]["fgsm"].append(
                acc_on_batch(model, fgsm_attack(model, X, y, epsilon), y))
            results[tag]["pgd"].append(
                acc_on_batch(model, pgd_attack(model, X, y, epsilon),  y))

    print(f"\n{'Model':<20} {'Clean':>8} {'FGSM':>8} {'PGD':>8}")
    print("-" * 48)
    for tag, r in results.items():
        print(f"{tag:<20} {np.mean(r['clean']):>8.4f} "
              f"{np.mean(r['fgsm']):>8.4f} {np.mean(r['pgd']):>8.4f}")

    tags   = list(results.keys())
    cleans = [np.mean(results[t]["clean"]) for t in tags]
    fgsms  = [np.mean(results[t]["fgsm"])  for t in tags]
    pgds   = [np.mean(results[t]["pgd"])   for t in tags]

    x = np.arange(len(tags)); w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, cleans, w, label="Clean")
    ax.bar(x,     fgsms,  w, label="FGSM")
    ax.bar(x + w, pgds,   w, label="PGD")
    ax.set_xticks(x); ax.set_xticklabels(tags)
    ax.set_ylabel("Accuracy"); ax.set_title("Adversarial Robustness Comparison")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "adversarial_robustness.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")
    return results


# ─────────────────────────────────────────────────────────────
# STEP 11 — Main Pipeline
# ─────────────────────────────────────────────────────────────

def main():

    # ── 11.1  Hyperparameter search ──────────────────────────
    best_mlp_params = run_optuna_mlp(n_trials=15)
    lr_mlp = best_mlp_params.get("lr", 1e-3)
    bs_mlp = best_mlp_params.get("batch_size", 128)
    dr_mlp = best_mlp_params.get("dropout_rate", 0.4)
    wd_mlp = best_mlp_params.get("weight_decay", 1e-4)

    # ── 11.2  Data loaders ───────────────────────────────────
    train_l, val_l, test_l = get_loaders(train_transform, test_transform, bs_mlp)
    pt_train, pt_val, pt_test = get_loaders(
        pretrained_transform_train, pretrained_transform_test, batch_size=64)

    # ── 11.3  Train MLP ──────────────────────────────────────
    print("\n=== Training Deep MLP ===")
    mlp = DeepMLP(dropout_rate=dr_mlp)
    mlp, hist_mlp = full_train(mlp, train_l, val_l,
                               epochs=80, lr=lr_mlp,
                               weight_decay=wd_mlp, patience=15, tag="MLP")

    # ── 11.4  Train CNN ──────────────────────────────────────
    print("\n=== Training CNN ===")
    cnn = CNN(dropout_rate=0.3)
    cnn, hist_cnn = full_train(cnn, train_l, val_l,
                               epochs=80, lr=5e-4,
                               weight_decay=1e-4, patience=15, tag="CNN")

    # ── 11.5  Fine-tune ResNet18 ─────────────────────────────
    print("\n=== Fine-tuning ResNet18 (Transfer Learning) ===")
    resnet = build_pretrained_resnet(fine_tune_layers=2)
    resnet, hist_resnet = full_train(resnet, pt_train, pt_val,
                                     epochs=30, lr=1e-3,
                                     weight_decay=1e-4, patience=8, tag="ResNet18")

    # ── 11.6  Evaluation ─────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    _, _, preds_mlp,    labels_mlp    = evaluate(mlp,    test_l,  criterion)
    _, _, preds_cnn,    labels_cnn    = evaluate(cnn,    test_l,  criterion)
    _, _, preds_resnet, labels_resnet = evaluate(resnet, pt_test, criterion)

    print_metrics(labels_mlp,    preds_mlp,    mlp,    test_l,  "MLP")
    print_metrics(labels_cnn,    preds_cnn,    cnn,    test_l,  "CNN")
    print_metrics(labels_resnet, preds_resnet, resnet, pt_test, "ResNet18")

    # ── 11.7  Training curves & confusion matrices ────────────
    plot_history({"MLP": hist_mlp, "CNN": hist_cnn, "ResNet18": hist_resnet})
    plot_confusion_matrix(labels_mlp,    preds_mlp,    "MLP")
    plot_confusion_matrix(labels_cnn,    preds_cnn,    "CNN")
    plot_confusion_matrix(labels_resnet, preds_resnet, "ResNet18")

    # ── 11.8  Grad-CAM ───────────────────────────────────────
    print("\n=== Grad-CAM Visualisations ===")
    visualise_gradcam(cnn,    cnn.features[-3],        test_l,  tag="CNN",      n=10)
    visualise_gradcam(resnet, resnet.layer4[-1].conv2, pt_test, tag="ResNet18", n=10)

    # ── 11.9  SHAP for MLP ───────────────────────────────────
    print("\n=== SHAP Explanation for MLP ===")
    explain_mlp_shap(mlp, test_l)

    # ── 11.10 Adversarial robustness ─────────────────────────
    robustness_eval({"MLP": mlp, "CNN": cnn}, test_l,  epsilon=0.03, n_batches=20)
    robustness_eval({"ResNet18": resnet},     pt_test, epsilon=0.03, n_batches=10)

    print("\n✓ All results saved to:", SAVE_DIR)


if __name__ == "__main__":
    main()
