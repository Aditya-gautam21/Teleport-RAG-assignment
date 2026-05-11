# Migrating to Vertex AI Vector Search (Matching Engine)

## Current State (Local Prototype)

The local RAG pipeline uses:
- **Embeddings**: `all-MiniLM-L6-v2` via HuggingFace (384 dimensions)
- **Vector Store**: FAISS in-memory, persisted to disk
- **Query Expansion**: Local Gemma 4B via llama-cpp

These are wrapped behind `MockTextEmbeddingModel` and `MockGenerativeModel`, which mirror the Vertex AI SDK signatures. The mock layer is designed so that swapping to real Vertex AI services requires changing only the imports, not the pipeline logic.

## Migration Steps

### Step 1: Replace the Embedding Backend

**Local (current):**
```python
self.embedder = MockTextEmbeddingModel(model_name="textembedding-gecko")
```

**Vertex AI (production):**
```python
from vertexai.language_models import TextEmbeddingModel

self.embedder = TextEmbeddingModel.from_pretrained("textembedding-gecko@latest")
```

The method signature is identical: `.get_embeddings(texts)` returns `TextEmbedding` objects with `.values`. The pipeline's `search_raw()` and `search_enhanced()` methods would work without any changes.

Consideration: Vertex AI's `textembedding-gecko` produces 768-dimensional vectors (vs. 384 for all-MiniLM-L6-v2). Re-indexing is required when switching the embedding model.

### Step 2: Replace the Query Expansion Backend

**Local (current):**
```python
self.expander = MockGenerativeModel(model_name="gemini-pro")
```

**Vertex AI (production):**
```python
from vertexai.language_models import GenerativeModel

self.expander = GenerativeModel("gemini-pro")
```

The `.predict(prompt, temperature=...)` method returns a `Prediction` object with `.text`, matching the mock exactly.

### Step 3: Migrate from FAISS to Vertex AI Vector Search

**Local (current):** FAISS in-memory index, saved to `data/vectorstore/index.faiss`.

**Production:** Use Vertex AI Vector Search (Matching Engine), which requires:

1. **Create an Index:**
```python
from google.cloud.aiplatform import MatchingEngineIndex

index = MatchingEngineIndex.create_tree_ah_index(
    display_name="rag-document-index",
    dimensions=768,
    distance_measure_type="DOT_PRODUCT_DISTANCE",
    shard_size="SHARD_SIZE_SMALL",
    approximate_neighbors_count=50,
)
```

2. **Create an Index Endpoint** for serving:
```python
endpoint = MatchingEngineIndexEndpoint.create(
    display_name="rag-serving-endpoint",
    public_endpoint_enabled=True,
)
endpoint.deploy_index(index=index, deployed_index_id="rag_index_v1")
```

3. **Batch Upload Embeddings:** Vector Search uses Cloud Storage as the ingestion path. Embeddings are written as JSONL files to a GCS bucket, then imported via `index.update(contents_uri="gs://...")`.

4. **Replace FAISS similarity_search:** The local `similarity_search_by_vector()` call is replaced with:
```python
response = endpoint.find_neighbors(
    deployed_index_id="rag_index_v1",
    queries=[query_vector],
    num_neighbors=k,
)
```

### Step 4: Key Production Considerations

| Concern | Local (FAISS) | Vertex AI Vector Search |
|---|---|---|
| Scaling | Single machine, memory-bound | Billions of vectors, auto-sharded |
| Availability | None (single process) | Multi-region, 99.95% SLA |
| Index updates | Rebuild from scratch | Incremental streaming updates |
| Latency | Sub-ms (in-memory) | Sub-100ms with ScaNN |
| Cost | Free (local compute) | Per GB-hour index + per-query pricing |
| Authentication | None | IAM / service accounts |
| Monitoring | None | Cloud Monitoring + Cloud Logging |

### Step 5: Code Changes Required

The `RAGPipeline` class would change minimally. The main difference is in `ingest()` — instead of `FAISS.from_documents()` + `save_local()`, embeddings would be written to GCS and imported into Vector Search. The `search_raw()` and `search_enhanced()` methods change only in the final search call — `endpoint.find_neighbors()` replaces `vector_store.similarity_search_by_vector()`.

The mock layer proves this migration is viable: both the local and production code paths share the same interface (`get_embeddings`, `predict`, vector search), so the orchestration class can be configured with either backend without rewriting the retrieval logic.
