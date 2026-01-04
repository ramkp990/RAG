import numpy as np
import pandas as pd
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
import pandas as pd
import numpy as np
from datetime import datetime

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