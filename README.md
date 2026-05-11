# RAG Pipeline with Vector Search — Benchmarking Raw vs AI-Enhanced Retrieval

A local Retrieval-Augmented Generation (RAG) pipeline that benchmarks two retrieval strategies: raw vector search (Strategy A) vs LLM-expanded query search (Strategy B). Built with FAISS, sentence-transformers, and a local LLM for query expansion. Vertex AI SDK interfaces are mocked to demonstrate GCP production readiness.

## Prerequisites

- Python 3.10+
- A GGUF-format LLM for query expansion (e.g., Gemma, Llama, Mistral)
- ~2 GB disk space for models and the FAISS index

## Setup

```bash
git clone <repo-url>
cd Teleport-RAG-assignment

pip install -r requirements.txt

cp .env.example .env
# Edit .env and set MODEL_PATH to your GGUF model file
```

### .env Configuration

```
MODEL_PATH='/absolute/path/to/your/model.gguf'
```

Any GGUF model that supports chat completion will work. Small models (2B–4B parameters, Q4 quantized) are sufficient for query expansion.

## Project Structure

```
├── docs/
│   ├── similarity_metrics.md       # Cosine vs Euclidean distance analysis
│   └── vertex_ai_migration.md      # FAISS → Vertex AI Vector Search migration plan
├── mocks/
│   ├── __init__.py
│   └── vertex_mocks.py             # MockTextEmbeddingModel, MockGenerativeModel
├── services/
│   ├── chunker.py                  # RecursiveCharacterTextSplitter
│   ├── data_loader.py              # Text file loader
│   ├── embeddings.py               # HuggingFace sentence-transformers wrapper
│   ├── pipeline.py                 # RAGPipeline orchestration class
│   └── vectorstore.py              # FAISS vector store (create, load, search)
├── testing/
│   ├── benchmark.py                # Strategy A vs B comparison runner
│   ├── expanded_search.py          # Strategy B standalone script
│   ├── raw_vector_search.py        # Strategy A standalone script
│   └── run_tests_json.py           # Pytest runner with JSON export
├── tests/
│   ├── test_mocks.py               # Mock class verification (10 tests)
│   └── test_pipeline.py            # RAGPipeline verification (19 tests)
├── utils/
│   └── prompts.py                  # Query expansion prompt template
├── sample_data.txt                 # 10 technical paragraphs (the corpus)
├── requirements.txt
├── .env.example
└── README.md
```

## Running the Pipeline

### 1. Run the Benchmark (main deliverable)

```bash
python testing/benchmark.py
```

This ingests `sample_data.txt`, runs 4 complex queries through both strategies, prints a side-by-side comparison table, and writes `retrieval_benchmark.md` with the JSON output.

### 2. Run Individual Searches

```bash
# Strategy A: raw vector search
python testing/raw_vector_search.py

# Strategy B: AI-enhanced search (query expansion via local LLM)
python testing/expanded_search.py
```

### 3. Run the Test Suite

```bash
# Standard pytest output
pytest tests/ -v

# With JSON report
python testing/run_tests_json.py
```

The JSON report is written to `test_results.json` at the project root.

## Architecture

```
User Query
    │
    ├── Strategy A ──► embed(raw_query) ──► FAISS similarity_search ──► results
    │
    └── Strategy B ──► LLM.expand(query) ──► embed(expanded) ──► FAISS search ──► results
```

Both strategies use the same FAISS index and embedding model (`all-MiniLM-L6-v2`, 384 dimensions). The only variable is whether the query passes through the query expander first.

The `MockTextEmbeddingModel` and `MockGenerativeModel` classes replicate the Vertex AI SDK interface (`vertexai.language_models`) while using local models under the hood, demonstrating readiness for migration to GCP's managed services.

## Distance Metric

This project uses **cosine similarity** (computed as inner product on L2-normalized vectors). See `docs/similarity_metrics.md` for a detailed comparison of cosine vs Euclidean distance and the rationale for this choice.
