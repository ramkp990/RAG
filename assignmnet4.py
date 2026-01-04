import pandas as pd
from sentence_transformers import SentenceTransformer
import hdbscan
import matplotlib.pyplot as plt
import umap
import numpy as np
import hdbscan
import pandas as pd
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from datetime import datetime

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



######  PART 2  ##########

from rank_bm25 import BM25Okapi
from sentence_transformers import util
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer

df = pd.read_csv(
    "combined_data_environment.csv",
    sep=";",
    encoding="latin1",     # handles emojis & special chars
    engine="python"        # safer for messy text
)
df["text"] = df["title"].fillna("") + ". " + df["content"]


# Prepare texts for BM25 (tokenized)
def tokenize(text):
    return text.lower().split()

df['text_for_embedding'] = df['title'].fillna('') + " " + df['content']
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(
    df['text_for_embedding'].tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)
print("✅ Embeddings shape:", embeddings.shape)  # (n_articles, 384)


print("Preparing BM25 index...")
corpus_texts = df['text'].fillna("").tolist()
tokenized_corpus = [tokenize(doc) for doc in corpus_texts]
bm25 = BM25Okapi(tokenized_corpus)


sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Hybrid Search Function 
def hybrid_search(query, top_k=5, alpha=0.6):
    """
    Hybrid search combining BM25 (lexical) and SBERT (semantic).

    """
    # BM25 Scores 
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)  # shape: (n_docs,)
    
    #  SBERT Scores 
    query_embedding = sbert_model.encode(query, convert_to_numpy=True)
    # Cosine similarity: [n_docs]
    sbert_scores = util.cos_sim(query_embedding, embeddings).cpu().numpy().flatten()
    
    # Normalize Scores to [0, 1] 
    scaler = MinMaxScaler()
    bm25_scores_norm = scaler.fit_transform(bm25_scores.reshape(-1, 1)).flatten()
    sbert_scores_norm = scaler.fit_transform(sbert_scores.reshape(-1, 1)).flatten()
    
    # Combine with alpha 
    hybrid_scores = alpha * sbert_scores_norm + (1 - alpha) * bm25_scores_norm
    
    # Get top-k indices 
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
    
    # Format results 
    results = []
    for idx in top_indices:
        results.append({
            'index': idx,
            'score': hybrid_scores[idx],
            'bm25_score': bm25_scores_norm[idx],
            'sbert_score': sbert_scores_norm[idx],
            'title': df.iloc[idx]['title'],
            'content': df.iloc[idx]['content'][:200] + "..."  # preview
        })
    return results

#  Demo Queries
demo_queries = [
    "What are green bonds?",
    "EU Taxonomy for sustainable activities",
    "How does carbon pricing work?",
    "ESG reporting requirements for companies"
]

print("\n" + "="*80)
print("TASK 1: HYBRID SEARCH DEMO")
print("="*80)

for query in demo_queries:
    print(f"\n🔍 Query: '{query}'")
    print("-" * 50)
    
    # Compare alpha=0.0 (BM25), 0.5 (balanced), 1.0 (SBERT)
    for alpha in [0.0, 0.5, 1.0]:
        results = hybrid_search(query, top_k=2, alpha=alpha)
        method = {0.0: "BM25 only", 0.5: "Balanced", 1.0: "SBERT only"}[alpha]
        print(f"\n[{method}] Top result:")
        print(f"  Title: {results[0]['title']}")
        print(f"  Preview: {results[0]['content']}")
    
    print("\n" + "="*50)



#  RERANKING WITH CROSS-ENCODER 
from sentence_transformers import CrossEncoder

#  Load Cross-Encoder (once) 
print("Loading Cross-Encoder for reranking...")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

#  Rerank Search Function
def rerank_search(query, top_k=3, candidates=20, alpha=0.6):
    """
    Two-stage retrieval: Hybrid search → Cross-encoder reranking.
    
    """
    # Stage 1: Retrieve candidates with hybrid search
    candidate_results = hybrid_search(query, top_k=candidates, alpha=alpha)
    candidate_texts = [df.iloc[res['index']]['text'] for res in candidate_results]
    
    # Stage 2: Rerank with cross-encoder
    # Format: list of (query, document) pairs
    pairs = [(query, doc) for doc in candidate_texts]
    ce_scores = cross_encoder.predict(pairs)  # shape: (candidates,)
    
    # Get top-k indices based on cross-encoder scores
    rerank_indices = np.argsort(ce_scores)[::-1][:top_k]
    
    # Build final results
    reranked_results = []
    for idx in rerank_indices:
        orig_result = candidate_results[idx]
        reranked_results.append({
            'index': orig_result['index'],
            'ce_score': ce_scores[idx],
            'hybrid_score': orig_result['score'],
            'title': orig_result['title'],
            'content': orig_result['content']
        })
    return reranked_results, candidate_results  # return both for comparison


print("\n" + "="*80)
print("TASK 2: RERANKING DEMO")
print("="*80)

test_query = "What are the key requirements of the EU Taxonomy?"
print(f"\n Query: '{test_query}'")

# Get results
reranked, original_candidates = rerank_search(test_query, top_k=2, candidates=10, alpha=0.6)

print("\n[BEFORE RERANKING] Top 2 from hybrid search:")
for i, res in enumerate(original_candidates[:2]):
    print(f"  {i+1}. [Score: {res['score']:.3f}] {res['title']}")

print("\n[AFTER RERANKING] Top 2 from cross-encoder:")
for i, res in enumerate(reranked):
    print(f"  {i+1}. [CE Score: {res['ce_score']:.3f}] {res['title']}")


# TEMPORAL SEARCH (BONUS)

#  Parse publication dates 
def parse_year(date_str):
    """
    Extract year from date string. Handles common formats in the dataset.
    Returns 2026 (current year) if parsing fails.
    """
    if pd.isna(date_str):
        return 2026
    try:
        # Try ISO format (e.g., "2023-05-17")
        return pd.to_datetime(date_str).year
    except:
        try:
            # Try other formats (e.g., "May 17, 2023")
            return pd.to_datetime(date_str, infer_datetime_format=True).year
        except:
            return 2026  # fallback to current year

# Add year column to df
print("Parsing publication dates...")
df['year'] = df['publication_date'].apply(parse_year)
print(f"Years range: {df['year'].min()} – {df['year'].max()}")

# Temporal Search Function 
def temporal_search(query, top_k=3, decay_rate=0.9, alpha=0.6, candidates=20):
    """
    Apply time decay to hybrid search results.

    """
    # Get hybrid search results
    results = hybrid_search(query, top_k=candidates, alpha=alpha)
    
    # Apply time decay
    current_year = 2026  # as per system time in prompt
    for res in results:
        doc_year = df.iloc[res['index']]['year']
        age = current_year - doc_year
        decay_weight = decay_rate ** age
        res['temporal_score'] = res['score'] * decay_weight
        res['decay_weight'] = decay_weight
    
    # Sort by temporal score
    results = sorted(results, key=lambda x: x['temporal_score'], reverse=True)
    return results[:top_k]

# Hard Filtering Function (for comparison) ---
def hard_filtered_search(query, top_k=3, min_year=2024, alpha=0.6):
    """Filter documents to only those >= min_year, then run hybrid search."""
    mask = df['year'] >= min_year
    filtered_indices = df[mask].index.tolist()
    
    # Get hybrid scores for all docs
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    query_embedding = sbert_model.encode(query, convert_to_numpy=True)
    sbert_scores = util.cos_sim(query_embedding, embeddings).cpu().numpy().flatten()
    
    scaler = MinMaxScaler()
    bm25_norm = scaler.fit_transform(bm25_scores.reshape(-1, 1)).flatten()
    sbert_norm = scaler.fit_transform(sbert_scores.reshape(-1, 1)).flatten()
    hybrid_scores = 0.6 * sbert_norm + 0.4 * bm25_norm
    
    # Mask and get top-k in filtered set
    masked_scores = np.full(len(df), -np.inf)
    masked_scores[filtered_indices] = hybrid_scores[filtered_indices]
    top_indices = np.argsort(masked_scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'index': idx,
            'score': hybrid_scores[idx],
            'title': df.iloc[idx]['title'],
            'year': df.iloc[idx]['year']
        })
    return results

# --- 4. Demo: Compare Temporal Strategies ---
print("\n" + "="*80)
print("TASK 3: TEMPORAL SEARCH DEMO")
print("="*80)

temporal_query = "Latest developments in carbon pricing policy"
print(f"\n🔍 Query: '{temporal_query}'")

# Standard search (no time handling)
standard_results = hybrid_search(temporal_query, top_k=2, alpha=0.6)
print("\n[STANDARD SEARCH] (ignores time)")
for i, res in enumerate(standard_results):
    year = df.iloc[res['index']]['year']
    print(f"  {i+1}. [{year}] {res['title']}")

# Hard filtering (last 2 years)
hard_results = hard_filtered_search(temporal_query, top_k=2, min_year=2024)
print("\n[HARD FILTERING] (2024–2026 only)")
for i, res in enumerate(hard_results):
    print(f"  {i+1}. [{hard_results[i]['year']}] {hard_results[i]['title']}")

# Time decay (decay_rate=0.9)
decay_results = temporal_search(temporal_query, top_k=2, decay_rate=0.9, alpha=0.6)
print("\n[TIME DECAY] (decay_rate=0.9)")
for i, res in enumerate(decay_results):
    year = df.iloc[res['index']]['year']
    weight = res['decay_weight']
    print(f"  {i+1}. [{year}, weight={weight:.2f}] {res['title']}")

# Compare decay rates
print("\n" + "-"*60)
print("Effect of different decay rates:")
for rate in [0.8, 0.9, 0.95]:
    results = temporal_search(temporal_query, top_k=1, decay_rate=rate, alpha=0.6)
    year = df.iloc[results[0]['index']]['year']
    print(f"  decay_rate={rate}: [{year}] {results[0]['title'][:60]}...")

# --- TASK 4: RAG WITH CITATIONS ---
from llama_cpp import Llama
import os

# --- 1. Initialize LLM (once) ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"

print("Initializing LLM (first run will download ~300MB)...")
llm = Llama.from_pretrained(
    repo_id=MODEL_NAME,
    filename=MODEL_FILE,
    n_ctx=2048,        # context window
    n_threads=4,       # adjust for your CPU
    verbose=False      # suppress download logs
)

# --- 2. Format Retrieved Docs for Prompt ---
def format_context(retrieved_results):
    """Format retrieved docs as [1] Title\nContent\n[2] ..."""
    context_parts = []
    for i, res in enumerate(retrieved_results):
        idx = i + 1
        title = df.iloc[res['index']]['title'] or "Untitled"
        content = df.iloc[res['index']]['content'] or ""
        # Truncate long content to fit context window
        content = content[:800]  # ~200 words
        context_parts.append(f"[{idx}] {title}\n{content}")
    return "\n\n".join(context_parts)

# --- 3. RAG Pipeline Function ---
def rag(query, top_k=3, candidates=20, alpha=0.6, max_tokens=256):
    """
    RAG pipeline with citation-aware generation.
    
    Returns:
        answer (str): Generated answer with [1], [2] citations or "Cannot answer"
    """
    # Retrieve top-k docs using reranker
    reranked_results, _ = rerank_search(query, top_k=top_k, candidates=candidates, alpha=alpha)
    
    # Format context with citations
    context = format_context(reranked_results)
    
    # Build prompt
    prompt = f"""Based ONLY on the following sources, answer the question below.
Use [N] to cite sources (e.g., "Green bonds finance climate projects [1].").
If the answer is not contained in the sources, respond with: "Cannot answer from sources."

Sources:
{context}

Question: {query}
Answer:"""
    
    # Generate response
    response = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.3,     
        stop=["\n", "Question:"], 
        echo=False
    )
    
    answer = response['choices'][0]['text'].strip()
    
    # Post-process: ensure unanswerable queries are handled
    if "cannot answer" in answer.lower() or "not mentioned" in answer.lower():
        return "Cannot answer from sources."
    
    return answer

# --- 4. Demo: Answerable vs Unanswerable Queries ---
print("\n" + "="*80)
print("TASK 4: RAG WITH CITATIONS DEMO")
print("="*80)

# Answerable query
answerable_query = "What are the main goals of the EU Taxonomy?"
print(f"\n Query: '{answerable_query}'")
answer = rag(answerable_query, top_k=2)
print(f" Answer: {answer}")

# Unanswerable query
unanswerable_query = "What is the stock price of Tesla today?"
print(f"\n Query: '{unanswerable_query}'")
answer = rag(unanswerable_query, top_k=2)
print(f" Answer: {answer}")

# Show retrieved context for transparency (optional in report)
print("\n" + "-"*60)
print("Retrieved context for first query:")
reranked, _ = rerank_search(answerable_query, top_k=2)
context = format_context(reranked)
print(context[:500] + "...")
llm.close()