import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import pdist
import umap.umap_ as umap
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. Dataset Loading & Preprocessing
# ==========================================
print("Fetching MNIST dataset (this may take a moment)...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

# Subsampling for computational feasibility (KPCA & distance matrices are O(N^2))
np.random.seed(42)
idx = np.random.choice(len(X), 6000, replace=False)
X_sub, y_sub = X[idx], y[idx]

# Normalize pixel values to [0, 1]
scaler = MinMaxScaler()
X_sub = scaler.fit_transform(X_sub)

X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size=1000, stratify=y_sub, random_state=42)

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}\n")

# ==========================================
# 2. Define Dimensionality Reduction Models
# ==========================================
dim = 2 # Reducing to 2D for visualization and k-NN

# 2a. PCA
print("Fitting PCA...")
pca = PCA(n_components=dim, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 2b. Kernel PCA
print("Fitting Kernel PCA (RBF)...")
kpca = KernelPCA(n_components=dim, kernel='rbf', fit_inverse_transform=True, random_state=42)
X_train_kpca = kpca.fit_transform(X_train)
X_test_kpca = kpca.transform(X_test)

# 2c. t-SNE (Grid Search Perplexity)
print("Tuning t-SNE...")
tsne_models = {}
for p in [5, 30, 50]:
    tsne = TSNE(n_components=dim, perplexity=p, random_state=42, init='pca', learning_rate='auto')
    tsne_models[p] = tsne.fit_transform(X_train) # t-SNE has no transform method for test sets

# We select Perplexity 30 as optimal for general datasets
X_train_tsne = tsne_models[30] 

# 2d. UMAP (Tuning n_neighbors and min_dist)
print("Tuning UMAP...")
# In a real scenario we'd cross-validate, here we pick a known strong config for MNIST
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=dim, random_state=42)
X_train_umap = umap_model.fit_transform(X_train)

# 2e. Undercomplete Autoencoder
print("Training Autoencoder...")
input_img = Input(shape=(784,))
encoded = Dense(256, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
bottleneck = Dense(dim, activation='linear', name='bottleneck')(encoded) # Bottleneck
decoded = Dense(64, activation='relu')(bottleneck)
decoded = Dense(256, activation='relu')(decoded)
output_img = Dense(784, activation='sigmoid')(decoded)

autoencoder = Model(input_img, output_img)
encoder = Model(input_img, bottleneck)

autoencoder.compile(optimizer='adam', loss='mse')
# Quiet training to keep console clean
autoencoder.fit(X_train, X_train, epochs=20, batch_size=256, validation_data=(X_test, X_test), verbose=0)

X_train_ae = encoder.predict(X_train, verbose=0)
X_test_ae = encoder.predict(X_test, verbose=0)

# ==========================================
# 3. Quantitative Evaluation
# ==========================================
print("\n--- Quantitative Evaluation ---")

# 3a. Reconstruction Error (MSE) on Test Set
pca_recon = mean_squared_error(X_test, pca.inverse_transform(X_test_pca))
kpca_recon = mean_squared_error(X_test, kpca.inverse_transform(X_test_kpca))
ae_recon = mean_squared_error(X_test, autoencoder.predict(X_test, verbose=0))

print(f"Reconstruction MSE | PCA: {pca_recon:.4f} | KPCA: {kpca_recon:.4f} | Autoencoder: {ae_recon:.4f}")

# 3b. Kruskal's Stress & Trustworthiness (Computed on a 1000-sample subset to prevent memory overflow)
subset_idx = np.random.choice(len(X_train), 1000, replace=False)
X_train_sub = X_train[subset_idx]
D_high = pdist(X_train_sub)

def compute_stress(X_high, X_low):
    D_low = pdist(X_low)
    # Scale D_low to match D_high scale to prevent artificial inflation of stress
    scale = np.sum(D_high * D_low) / np.sum(D_low ** 2)
    D_low_scaled = D_low * scale
    return np.sqrt(np.sum((D_high - D_low_scaled)**2) / np.sum(D_high**2))

print("\nManifold Metrics (Subset):")
tsne_sub = X_train_tsne[subset_idx]
umap_sub = X_train_umap[subset_idx]

print(f"t-SNE (p=30) -> Trustworthiness: {trustworthiness(X_train_sub, tsne_sub):.4f} | Kruskal Stress: {compute_stress(X_train_sub, tsne_sub):.4f}")
print(f"UMAP         -> Trustworthiness: {trustworthiness(X_train_sub, umap_sub):.4f} | Kruskal Stress: {compute_stress(X_train_sub, umap_sub):.4f}")

# 3c. Downstream Classification (5-Fold CV with k-NN)
knn = KNeighborsClassifier(n_neighbors=5)
embeddings = {
    'PCA': X_train_pca,
    'Kernel PCA': X_train_kpca,
    't-SNE': X_train_tsne,
    'UMAP': X_train_umap,
    'Autoencoder': X_train_ae
}

print("\nDownstream k-NN (k=5) 5-Fold CV Accuracy:")
for name, emb in embeddings.items():
    acc = cross_val_score(knn, emb, y_train, cv=5).mean()
    print(f"{name:>15}: {acc:.4f}")

# ==========================================
# 4. Visualizations
# ==========================================
# Plot 1: 2D Embeddings Comparison
fig, axes = plt.subplots(1, 5, figsize=(25, 5))
fig.suptitle('2D Embeddings of MNIST via Different Methods', fontsize=16)

y_train_int = y_train.astype(int)
cmap = plt.get_cmap('tab10')

for i, (name, emb) in enumerate(embeddings.items()):
    scatter = axes[i].scatter(emb[:, 0], emb[:, 1], c=y_train_int, cmap=cmap, s=2, alpha=0.7)
    axes[i].set_title(name)
    axes[i].set_xticks([])
    axes[i].set_yticks([])

legend1 = axes[-1].legend(*scatter.legend_elements(), loc="lower right", title="Digits")
axes[-1].add_artist(legend1)
plt.tight_layout()
plt.show()

# Plot 2: Autoencoder Latent Space Detail
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_train_ae[:, 0], X_train_ae[:, 1], c=y_train_int, cmap='tab10', s=5, alpha=0.8)
plt.colorbar(scatter, label='Digit Class')
plt.title("Autoencoder 2D Latent Space Semantic Structure")
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.grid(True, alpha=0.3)
plt.show()