import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, davies_bouldin_score,
                             adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score)
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import umap.umap_ as umap
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. Dataset Loading & Preprocessing
# ==========================================
print("Loading Optdigits dataset...")
digits = load_digits()
X = digits.data
y_true = digits.target # Used ONLY for external validation, NOT for training
n_true_classes = len(np.unique(y_true)) # 10 digits

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Dataset shape: {X_scaled.shape}\n")

# ==========================================
# 2. Hyperparameter Tuning & Diagnostics
# ==========================================
print("Running hyperparameter diagnostics...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Clustering Hyperparameter Diagnostics', fontsize=16)

# a) K-Means Elbow & Silhouette
k_range = range(2, 15)
inertias, sil_scores = [], []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

ax1 = axes[0, 0]
ax1.plot(k_range, inertias, marker='o', color='b', label='Inertia (Elbow)')
ax1.set_xlabel('Number of clusters (k)')
ax1.set_ylabel('Inertia')
ax1_2 = ax1.twinx()
ax1_2.plot(k_range, sil_scores, marker='s', color='r', label='Silhouette Score')
ax1_2.set_ylabel('Silhouette Score')
ax1.set_title('K-Means: Elbow & Silhouette')

# b) GMM BIC/AIC
bics, aics = [], []
for k in k_range:
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42).fit(X_scaled)
    bics.append(gmm.bic(X_scaled))
    aics.append(gmm.aic(X_scaled))

axes[0, 1].plot(k_range, bics, marker='o', label='BIC')
axes[0, 1].plot(k_range, aics, marker='s', label='AIC')
axes[0, 1].set_xlabel('Number of components')
axes[0, 1].set_ylabel('Information Criterion')
axes[0, 1].set_title('GMM: BIC / AIC')
axes[0, 1].legend()

# c) DBSCAN k-distance graph (to find eps)
nearest_neighbors = NearestNeighbors(n_neighbors=5) # min_samples ≈ dimensionality or rule of thumb
neighbors = nearest_neighbors.fit(X_scaled)
distances, indices = neighbors.kneighbors(X_scaled)
distances = np.sort(distances[:, 4], axis=0) # Sort 5th nearest neighbor distances

axes[1, 0].plot(distances)
axes[1, 0].set_xlabel('Points sorted by distance')
axes[1, 0].set_ylabel('5-NN distance')
axes[1, 0].set_title('DBSCAN: k-distance Graph')
axes[1, 0].axhline(y=4.5, color='r', linestyle='--', label='Chosen Eps (approx elbow)')
axes[1, 0].legend()

# d) Agglomerative Dendrogram (using a subset for visual clarity)
Z = linkage(X_scaled[:200], method='ward')
dendrogram(Z, ax=axes[1, 1], truncate_mode='level', p=5)
axes[1, 1].set_title('Agglomerative: Ward Dendrogram (Subset)')
plt.tight_layout()
plt.show()

# ==========================================
# 3. Fit Models (using optimal parameters identified above)
# ==========================================
k_opt = 10 # Known from data, supported by metrics
print(f"\nFitting models (k={k_opt})...")

models = {
    'K-Means': KMeans(n_clusters=k_opt, random_state=42, n_init=10),
    'GMM': GaussianMixture(n_components=k_opt, random_state=42),
    'DBSCAN': DBSCAN(eps=4.5, min_samples=5),
    'Agglomerative': AgglomerativeClustering(n_clusters=k_opt, linkage='ward')
}

labels_dict = {}
for name, model in models.items():
    if name == 'GMM':
        labels_dict[name] = model.fit_predict(X_scaled)
    else:
        labels_dict[name] = model.fit_predict(X_scaled)

# ==========================================
# 4. Cluster Ensemble (Co-Association Matrix)
# ==========================================
print("Building Cluster Ensemble (Co-association matrix)...")
n_samples = X_scaled.shape[0]
co_assoc_matrix = np.zeros((n_samples, n_samples))
ensemble_models = ['K-Means', 'GMM', 'Agglomerative'] # Exclude DBSCAN due to noise (-1) points complicating consensus

for name in ensemble_models:
    labels = labels_dict[name]
    # Create connectivity matrix for this model
    connectivity = (labels[:, None] == labels[None, :]).astype(int)
    co_assoc_matrix += connectivity

co_assoc_matrix /= len(ensemble_models)

# Convert similarity (co-association) to distance
distance_matrix = 1.0 - co_assoc_matrix
# Ensure diagonal is exactly 0
np.fill_diagonal(distance_matrix, 0)
# Condense distance matrix for scipy linkage
condensed_dist = squareform(distance_matrix, checks=False)

# Final ensemble clustering using Ward linkage on the distance matrix
ensemble_linkage = linkage(condensed_dist, method='ward')
labels_dict['Ensemble'] = fcluster(ensemble_linkage, k_opt, criterion='maxclust')

# ==========================================
# 5. Evaluation Metrics
# ==========================================
print("\n--- Evaluation Metrics ---")
results = []

def evaluate_clusters(name, labels):
    # Filter out noise points (-1) for DBSCAN internal metrics
    mask = labels != -1
    X_masked = X_scaled[mask]
    labels_masked = labels[mask]
    
    if len(np.unique(labels_masked)) > 1:
        sil = silhouette_score(X_masked, labels_masked)
        ch = calinski_harabasz_score(X_masked, labels_masked)
        db = davies_bouldin_score(X_masked, labels_masked)
    else:
        sil, ch, db = np.nan, np.nan, np.nan
        
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    fmi = fowlkes_mallows_score(y_true, labels)
    
    return {'Model': name, 'Silhouette': sil, 'Calinski-Harabasz': ch, 
            'Davies-Bouldin': db, 'ARI': ari, 'NMI': nmi, 'FMI': fmi}

for name, labels in labels_dict.items():
    results.append(evaluate_clusters(name, labels))

results_df = pd.DataFrame(results).set_index('Model')
print(results_df.round(3).to_string())

# ==========================================
# 6. Cluster Stability Analysis (Bootstrap)
# ==========================================
print("\n--- Cluster Stability Analysis (80% Bootstrap) ---")
n_iterations = 10
stability_scores = {'K-Means': [], 'GMM': [], 'Agglomerative': []}

for i in range(n_iterations):
    # Subsample 80% of data
    X_boot, y_boot = resample(X_scaled, y_true, n_samples=int(0.8 * n_samples), random_state=i)
    
    for name in stability_scores.keys():
        if name == 'GMM':
            model = GaussianMixture(n_components=k_opt, random_state=i)
        elif name == 'K-Means':
            model = KMeans(n_clusters=k_opt, random_state=i, n_init=5)
        else:
            model = AgglomerativeClustering(n_clusters=k_opt)
            
        boot_labels = model.fit_predict(X_boot)
        # Compare to ground truth of the bootstrap sample to measure stable structural recovery
        stability_scores[name].append(adjusted_rand_score(y_boot, boot_labels))

for name, scores in stability_scores.items():
    print(f"{name} Stability (Mean ARI) across {n_iterations} iterations: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

# ==========================================
# 7. Visualization (UMAP Projections)
# ==========================================
print("\nGenerating UMAP Projections...")
umap_reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_reducer.fit_transform(X_scaled)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('UMAP Projections of Cluster Assignments', fontsize=16)
axes = axes.flatten()

# Plot Ground Truth
scatter = axes[0].scatter(X_umap[:, 0], X_umap[:, 1], c=y_true, cmap='tab10', s=10)
axes[0].set_title('Ground Truth Labels')
axes[0].set_xticks([]); axes[0].set_yticks([])

# Plot Models
for i, (name, labels) in enumerate(labels_dict.items(), start=1):
    if i < len(axes):
        # DBSCAN noise points (-1) in black
        cmap = 'tab10' if name != 'DBSCAN' else 'nipy_spectral'
        scatter = axes[i].scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap=cmap, s=10)
        axes[i].set_title(name)
        axes[i].set_xticks([]); axes[i].set_yticks([])

plt.tight_layout()
plt.show()