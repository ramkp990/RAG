import pandas as pd
from sentence_transformers import SentenceTransformer
import hdbscan
import matplotlib.pyplot as plt
import umap
import numpy as np
import hdbscan
import pandas as pd
import seaborn as sns

df = pd.read_csv(
    "combined_data_environment.csv",
    sep=";",
    encoding="latin1",     # handles emojis & special chars
    engine="python"        # safer for messy text
)

print("Loaded rows:", len(df))
print(df.columns)
# Keep only rows with content
df = df.dropna(subset=["content"])

# Combine title + content (recommended)
df["text"] = df["title"].fillna("") + ". " + df["content"]

print(df["text"].iloc[0][:500])


from sentence_transformers import SentenceTransformer
import umap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors


# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')
df['text_for_embedding'] = df['title'].fillna('') + " " + df['content']

# Generate embeddings
embeddings = model.encode(
    df['text_for_embedding'].tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)
print("✅ Embeddings shape:", embeddings.shape)  # (n_articles, 384)

# UMAP: 50D for clustering
print("Running UMAP 50D...")
umap_50 = umap.UMAP(n_neighbors=30, min_dist=0.0, metric='cosine', n_components=50, random_state=42)
embeddings_50d = umap_50.fit_transform(embeddings)

# UMAP: 2D for visualization
print("Running UMAP 2D...")
umap_2 = umap.UMAP(n_neighbors=20, min_dist=0.1, metric='cosine', n_components=2, random_state=42)
embeddings_2d = umap_2.fit_transform(embeddings)

# Save for later use (optional but recommended)
np.save("embeddings_50d.npy", embeddings_50d)
np.save("embeddings_2d.npy", embeddings_2d)

# Visualize 2D
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=5, alpha=0.6)
plt.title("2D UMAP Projection of Environmental News Articles")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.tight_layout()
plt.show()

# Set min_samples 
min_samples = 10  

# Compute core distances in original embedding space 
# Use sklearn's NearestNeighbors to get distances to k=min_samples neighbors
nbrs = NearestNeighbors(n_neighbors=min_samples, metric='euclidean', algorithm='brute')
nbrs.fit(embeddings_50d)

# distances[:, -1] is distance to the (min_samples)-th neighbor (0-indexed, so index = min_samples-1)
distances, _ = nbrs.kneighbors(embeddings_50d)
core_distances = distances[:, -1]  # shape: (n_points,)

print("Core distances computed. Range:", core_distances.min(), "to", core_distances.max())

#  Plot heatmap 
plt.figure(figsize=(10, 8))

# Use reversed colormap: smaller (denser) = darker
scatter = plt.scatter(
    embeddings_2d[:, 0],
    embeddings_2d[:, 1],
    c=core_distances,
    cmap='viridis_r',   # _r = reversed → small values = dark/blue
    s=5,
    alpha=0.8
)

plt.colorbar(scatter, label='Core Distance (cosine)')
plt.title(f'Core Distances (min_samples = {min_samples}) on 2D UMAP')
np.save("core_distances.npy", core_distances)
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.tight_layout()
plt.show()


nonzero_mask = core_distances > 1e-8
if not np.any(nonzero_mask):
    # Fallback: use a random point
    focal_idx = np.random.choice(len(core_distances))
else:
    # Pick the point with smallest *non-zero* core distance
    candidate_indices = np.where(nonzero_mask)[0]
    focal_idx = candidate_indices[np.argmin(core_distances[nonzero_mask])]

print(f"Focal point index: {focal_idx}, core distance: {core_distances[focal_idx]:.6f}")

# --- Find 15 nearest neighbors (excluding self) ---
n_neighbors = 15
nbrs_full = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='euclidean', algorithm='brute')
nbrs_full.fit(embeddings_50d)

#distances_focal, indices_focal = nbrs_full.kneighbors(embeddings[focal_idx].reshape(1, -1))
#neighbor_indices = indices_focal[0][1:]  # exclude self
#neighbor_distances = distances_focal[0][1:]
focal_point_50d = embeddings_50d[focal_idx].reshape(1, -1)
distances_focal, indices_focal = nbrs_full.kneighbors(focal_point_50d)

neighbor_indices = indices_focal[0][1:]
neighbor_distances = distances_focal[0][1:] 

# --- Compute mutual reachability ---
core_focal = core_distances[focal_idx]
core_neighbors = core_distances[neighbor_indices]

mutual_reach_dists = np.maximum.reduce([
    np.full_like(neighbor_distances, core_focal),
    core_neighbors,
    neighbor_distances
])

print("Mutual reachability range:", mutual_reach_dists.min(), mutual_reach_dists.max())

# --- Handle edge case: all distances same ---
if mutual_reach_dists.max() == mutual_reach_dists.min():
    # Add tiny variation for visualization (or use uniform color)
    colors = np.full_like(mutual_reach_dists, 0.5)  # mid-gray
    use_uniform = True
else:
    colors = (mutual_reach_dists - mutual_reach_dists.min()) / (mutual_reach_dists.max() - mutual_reach_dists.min())
    use_uniform = False

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 8))

# Background points
ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=2, color='lightgray', alpha=0.5)

# Focal point
ax.scatter(
    embeddings_2d[focal_idx, 0],
    embeddings_2d[focal_idx, 1],
    s=200, color='red', marker='x', label='Focal Point'
)

# Neighbors
ax.scatter(
    embeddings_2d[neighbor_indices, 0],
    embeddings_2d[neighbor_indices, 1],
    s=50, color='blue', label='Neighbors'
)

# Edges
for i, neighbor_idx in enumerate(neighbor_indices):
    x_vals = [embeddings_2d[focal_idx, 0], embeddings_2d[neighbor_idx, 0]]
    y_vals = [embeddings_2d[focal_idx, 1], embeddings_2d[neighbor_idx, 1]]
    
    if use_uniform:
        edge_color = 'purple'  # or any single color
    else:
        edge_color = plt.cm.viridis(colors[i])
    
    ax.plot(x_vals, y_vals, color=edge_color, linewidth=2, alpha=0.8)

# Colorbar (only if not uniform)
if not use_uniform:
    sm = plt.cm.ScalarMappable(
        cmap='viridis',
        norm=plt.Normalize(vmin=mutual_reach_dists.min(), vmax=mutual_reach_dists.max())
    )
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Mutual Reachability Distance')

ax.set_title(f"Focal Point ({focal_idx}) and Neighbors\nCore dist: {core_focal:.4f}")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.legend()
plt.tight_layout()
plt.show()

min_cluster_sizes = [5, 10, 15, 20, 25, 30, 40]
min_samples_values = [5, 10, 15, 20]

results = []

for min_clust in min_cluster_sizes:
    for min_samp in min_samples_values:
        if min_samp > min_clust:
            continue  # invalid combo
            
        # Fit HDBSCAN
        clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_clust,
        min_samples=min_samp,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True,
        gen_min_span_tree=True,  
        # memory="hdbscan_cache",  
        algorithm='best'          
    )
        cluster_labels = clusterer.fit_predict(embeddings_50d)
        probabilities = clusterer.probabilities_
        
        # Metrics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        noise_ratio = np.sum(cluster_labels == -1) / len(cluster_labels)
        dbcv = clusterer.relative_validity_  # DBCV score (higher = better)
        mean_stability = np.mean(probabilities[cluster_labels != -1]) if n_clusters > 0 else 0.0
        
        results.append({
            'min_cluster_size': min_clust,
            'min_samples': min_samp,
            'n_clusters': n_clusters,
            'noise_ratio': noise_ratio,
            'dbcv': dbcv,
            'mean_stability': mean_stability
        })

# Convert to DataFrame
results_df = pd.DataFrame(results)
print(results_df.head())

# Pivot for heatmap
dbcv_pivot = results_df.pivot(
    index='min_samples',
    columns='min_cluster_size',
    values='dbcv'
)

plt.figure(figsize=(10, 6))
sns.heatmap(dbcv_pivot, annot=True, fmt=".3f", cmap="viridis")
plt.title("DBCV Score Across Hyperparameters")
plt.ylabel("min_samples")
plt.xlabel("min_cluster_size")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
scatter = plt.scatter(
    results_df['n_clusters'],
    results_df['noise_ratio'],
    c=results_df['dbcv'],
    cmap='viridis',
    s=100
)
plt.colorbar(scatter, label='DBCV')
plt.xlabel('Number of Clusters')
plt.ylabel('Noise Ratio')
plt.title('Trade-off: Clusters vs Noise (color = DBCV)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Normalize metrics (min-max)
def normalize(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-8)

# Higher is better for all, except noise_ratio (lower is better)
results_df['dbcv_norm'] = normalize(results_df['dbcv'])
results_df['stability_norm'] = normalize(results_df['mean_stability'])
results_df['noise_norm'] = 1 - normalize(results_df['noise_ratio'])  # invert
results_df['clusters_norm'] = normalize(results_df['n_clusters'].clip(0, 10))  # cap

# Weights (adjust based on priority)
w_dbcv = 0.4
w_stab = 0.3
w_noise = 0.2
w_clust = 0.1

results_df['composite_score'] = (
    w_dbcv * results_df['dbcv_norm'] +
    w_stab * results_df['stability_norm'] +
    w_noise * results_df['noise_norm'] +
    w_clust * results_df['clusters_norm']
)

# Get best
best_row = results_df.loc[results_df['composite_score'].idxmax()]
print("Best configuration:")
print(best_row[['min_cluster_size', 'min_samples', 'n_clusters', 'noise_ratio', 'dbcv', 'mean_stability']])


# Get best params from Task 4 (example)
best_min_clust = int(best_row['min_cluster_size'])
best_min_samp = int(best_row['min_samples'])

print(f"Fitting HDBSCAN with min_cluster_size={best_min_clust}, min_samples={best_min_samp}")

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=best_min_clust,
    min_samples=best_min_samp,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True,
    gen_min_span_tree=True,      
    algorithm='best'
)

cluster_labels = clusterer.fit_predict(embeddings_50d)

# Extract MST as (from, to, distance) array
mst = clusterer.minimum_spanning_tree_.to_numpy()
print("MST shape:", mst.shape)  # (n_points - 1, 3)

# Choose one non-noise cluster to visualize (e.g., largest cluster)
unique, counts = np.unique(cluster_labels, return_counts=True)
cluster_info = dict(zip(unique, counts))
# Remove noise (-1)
if -1 in cluster_info:
    del cluster_info[-1]
if not cluster_info:
    raise ValueError("No clusters found!")

# Pick largest cluster
target_cluster_id = max(cluster_info, key=cluster_info.get)
target_mask = cluster_labels == target_cluster_id
target_indices = np.where(target_mask)[0]
target_set = set(target_indices)

# Filter MST edges where BOTH endpoints are in target cluster
mst_in_cluster = []
for u, v, dist in mst:
    u, v = int(u), int(v)
    if u in target_set and v in target_set:
        mst_in_cluster.append((u, v, dist))

mst_in_cluster = np.array(mst_in_cluster)
print(f"Edges in cluster {target_cluster_id}: {len(mst_in_cluster)}")

# Plot
plt.figure(figsize=(12, 10))

# Background: all points
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=1, color='lightgray', alpha=0.5)

# Highlight target cluster
plt.scatter(
    embeddings_2d[target_mask, 0],
    embeddings_2d[target_mask, 1],
    s=10, color='steelblue', label=f'Cluster {target_cluster_id}'
)

# Draw MST edges

for u, v, dist in mst_in_cluster:
    u = int(u)
    v = int(v)
    x_vals = [embeddings_2d[u, 0], embeddings_2d[v, 0]]
    y_vals = [embeddings_2d[u, 1], embeddings_2d[v, 1]]
    
    # Normalize distance for color/thickness
    norm_dist = (dist - mst_in_cluster[:, 2].min()) / (mst_in_cluster[:, 2].max() - mst_in_cluster[:, 2].min() + 1e-8)
    color = plt.cm.viridis(1 - norm_dist)
    linewidth = 2 * (1 - norm_dist) + 0.5
    
    plt.plot(x_vals, y_vals, color=color, linewidth=linewidth, alpha=0.8)

plt.title(f"Minimum Spanning Tree (MST) for Cluster {target_cluster_id}\n"
          f"(Edges colored by mutual reachability distance)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend()
plt.tight_layout()
plt.show()

# Plot condensed tree
plt.figure(figsize=(14, 6))
clusterer.condensed_tree_.plot()
plt.title("HDBSCAN Condensed Tree")
plt.tight_layout()
plt.show()


#condensed_tree_df = clusterer.condensed_tree_.to_numpy()
# Format: [parent, child, lambda_birth, child_size]
#print("Condensed tree sample (parent, child, λ_birth, size):")
#print(condensed_tree_df[:5])

# Extract cluster persistence (already computed by HDBSCAN)
persistence = clusterer.cluster_persistence_
n_clusters_final = len(persistence)

if n_clusters_final == 0:
    print("No clusters selected!")
else:
    cluster_ids = np.unique(cluster_labels)
    cluster_ids = cluster_ids[cluster_ids != -1]  # exclude noise

    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(len(persistence)), persistence, color='skyblue')
    plt.xlabel('Cluster ID')
    plt.ylabel('Persistence (λ_birth - λ_death)')
    plt.title('Cluster Persistence (Excess of Mass Selection)')
    plt.xticks(range(len(persistence)), cluster_ids)
    
    # Annotate values
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{persistence[i]:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()



