# Similarity Metric Choice: Cosine vs. Euclidean Distance

## Why Cosine Similarity

This project uses **cosine similarity** as the distance metric for semantic search, which is the default in FAISS when using inner product on normalized vectors.

The choice is driven by how text embeddings work:

### 1. Direction > Magnitude in Semantic Space

Embedding models like `all-MiniLM-L6-v2` encode meaning primarily in the **direction** of the vector, not its length. Two vectors pointing in roughly the same direction represent semantically similar text, even if one is much longer than the other. Cosine similarity measures exactly this — the angle between vectors, ignoring magnitude.

A document chunk with 300 words and a query with 8 words will have very different vector magnitudes even when they're about the same topic. Cosine similarity handles this naturally. Euclidean distance would penalize the magnitude difference and push the longer chunk farther away, even if it's the most relevant result.

### 2. High-Dimensional Space Behavior

In high-dimensional spaces (384 dimensions in this project), Euclidean distance suffers from the "curse of dimensionality" — the contrast between nearest and farthest neighbors diminishes as dimensions increase. Cosine similarity is more robust in high dimensions because it only depends on the angular separation, which remains discriminative even as dimensionality grows.

### 3. FAISS Inner Product = Cosine on Normalized Vectors

FAISS uses inner product (dot product) as its default metric. For L2-normalized vectors (which sentence-transformers produce by default), inner product is mathematically equivalent to cosine similarity:

```
cosine_sim(A, B) = dot_product(A_norm, B_norm)
```

This means FAISS can use its highly optimized inner product kernels while getting cosine similarity semantics — no separate cosine implementation needed.

## When Euclidean Distance Would Be Better

Euclidean distance is preferable when:
- Vector magnitude carries meaningful information (e.g., TF-IDF weighted embeddings where longer documents genuinely match more query terms)
- The embedding model was explicitly trained with a Euclidean loss function
- You need a proper distance metric (satisfying the triangle inequality) for clustering or nearest-neighbor radius queries

## Comparison Table

| Property | Cosine Similarity | Euclidean Distance |
|---|---|---|
| Sensitive to magnitude | No | Yes |
| Range | [-1, 1] (normalized: [0, 1]) | [0, infinity) |
| Best for text embeddings | Yes | Rarely |
| FAISS optimization | Inner product kernel | L2 kernel |
| Curse of dimensionality | Less affected | More affected |
