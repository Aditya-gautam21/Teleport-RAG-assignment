# Retrieval Benchmark: Strategy A vs Strategy B

## Methodology

Strategy A embeds the raw user query directly and performs FAISS similarity search. Strategy B first expands the query using a local LLM (Gemma 4B) to add domain-specific keywords and alternative phrasings, then embeds the expanded query and searches the same index.

## Results

```json
[
  {
    "query": "How can external knowledge bases prevent language models from making up false information?",
    "strategy_a": {
      "query_used": "How can external knowledge bases prevent language models from making up false information?",
      "results": [
        "those documents as context into the language model's prompt. This approach significantly reduces hallucinations \u2014 instances where the model generates plausible but incorrect information \u2014 by anchoring outputs to verifiable source material.",
        "noisier embeddings that may dilute semantic focus. Smaller chunks capture precise concepts but risk fragmenting related information across multiple chunks. The optimal chunk size depends on the embedding model's context window, the nature of the documents, and the types of queries expected in production.",
        "models, including sentence-transformers like all-MiniLM-L6-v2 and Google's textembedding-gecko, are trained on massive corpora to capture nuanced semantic relationships, making them foundational components of any retrieval-augmented generation system."
      ]
    },
    "strategy_b": {
      "query_used": "User query: 'best vector database for semantic search'",
      "results": [
        "Hybrid search combines the strengths of vector similarity search with traditional keyword-based retrieval using techniques like reciprocal rank fusion (RRF). While semantic search excels at understanding conceptual similarity, it can miss exact keyword matches for proper nouns, product codes, or rare terms that appear infrequently in training data. A hybrid system runs both a vector search and a",
        "A hybrid system runs both a vector search and a lexical search (such as BM25) in parallel, then merges the result sets using RRF to produce a final ranking that balances semantic relevance with keyword precision. This approach consistently outperforms either method alone in production information retrieval benchmarks.",
        "Neighbors), developed by Google Research, use space-partitioning trees, quantization, and re-ranking stages to dramatically reduce the number of distance computations while maintaining high recall rates above 95%. Google's Vertex AI Vector Search is built on ScaNN technology and delivers low-latency similarity search for enterprise-scale deployments."
      ]
    }
  },
  {
    "query": "What are the tradeoffs between exact and approximate nearest neighbor search for large-scale vector retrieval?",
    "strategy_a": {
      "query_used": "What are the tradeoffs between exact and approximate nearest neighbor search for large-scale vector retrieval?",
      "results": [
        "Approximate nearest neighbor (ANN) algorithms are essential for scaling vector search to production workloads. Exact KNN search requires computing distances between the query vector and every vector in the database, which becomes prohibitively expensive as the dataset grows beyond millions of points. ANN algorithms like ScaNN (Scalable Nearest Neighbors), developed by Google Research, use",
        "Neighbors), developed by Google Research, use space-partitioning trees, quantization, and re-ranking stages to dramatically reduce the number of distance computations while maintaining high recall rates above 95%. Google's Vertex AI Vector Search is built on ScaNN technology and delivers low-latency similarity search for enterprise-scale deployments.",
        "The embedding dimensionality presents a practical tradeoff: higher-dimensional vectors (768 or 1024 dimensions) capture more semantic nuance but require more memory and slower comparison operations, while lower-dimensional vectors (384 dimensions) are faster to search and cheaper to store but may miss subtle semantic distinctions in complex queries."
      ]
    },
    "strategy_b": {
      "query_used": "User query: 'best vector database for semantic search'",
      "results": [
        "Hybrid search combines the strengths of vector similarity search with traditional keyword-based retrieval using techniques like reciprocal rank fusion (RRF). While semantic search excels at understanding conceptual similarity, it can miss exact keyword matches for proper nouns, product codes, or rare terms that appear infrequently in training data. A hybrid system runs both a vector search and a",
        "A hybrid system runs both a vector search and a lexical search (such as BM25) in parallel, then merges the result sets using RRF to produce a final ranking that balances semantic relevance with keyword precision. This approach consistently outperforms either method alone in production information retrieval benchmarks.",
        "Neighbors), developed by Google Research, use space-partitioning trees, quantization, and re-ranking stages to dramatically reduce the number of distance computations while maintaining high recall rates above 95%. Google's Vertex AI Vector Search is built on ScaNN technology and delivers low-latency similarity search for enterprise-scale deployments."
      ]
    }
  },
  {
    "query": "How does chunk size and overlap affect the quality of retrieved documents in a RAG pipeline?",
    "strategy_a": {
      "query_used": "How does chunk size and overlap affect the quality of retrieved documents in a RAG pipeline?",
      "results": [
        "Chunking is a critical preprocessing step in any RAG pipeline that directly impacts retrieval quality and downstream generation accuracy. The RecursiveCharacterTextSplitter algorithm splits documents into overlapping chunks based on character boundaries, with configurable chunk size and overlap parameters. Larger chunks preserve more context but produce noisier embeddings that may dilute semantic",
        "noisier embeddings that may dilute semantic focus. Smaller chunks capture precise concepts but risk fragmenting related information across multiple chunks. The optimal chunk size depends on the embedding model's context window, the nature of the documents, and the types of queries expected in production.",
        "Retrieval-augmented generation (RAG) is an architectural pattern that combines information retrieval with large language models to produce grounded, factually accurate responses. Instead of relying solely on parametric knowledge stored in model weights, a RAG pipeline first retrieves relevant documents from an external knowledge base, then feeds those documents as context into the language"
      ]
    },
    "strategy_b": {
      "query_used": "User query: 'best vector database for semantic search'",
      "results": [
        "Hybrid search combines the strengths of vector similarity search with traditional keyword-based retrieval using techniques like reciprocal rank fusion (RRF). While semantic search excels at understanding conceptual similarity, it can miss exact keyword matches for proper nouns, product codes, or rare terms that appear infrequently in training data. A hybrid system runs both a vector search and a",
        "A hybrid system runs both a vector search and a lexical search (such as BM25) in parallel, then merges the result sets using RRF to produce a final ranking that balances semantic relevance with keyword precision. This approach consistently outperforms either method alone in production information retrieval benchmarks.",
        "Neighbors), developed by Google Research, use space-partitioning trees, quantization, and re-ranking stages to dramatically reduce the number of distance computations while maintaining high recall rates above 95%. Google's Vertex AI Vector Search is built on ScaNN technology and delivers low-latency similarity search for enterprise-scale deployments."
      ]
    }
  },
  {
    "query": "When should a team choose cosine similarity over Euclidean distance for embedding-based search?",
    "strategy_a": {
      "query_used": "When should a team choose cosine similarity over Euclidean distance for embedding-based search?",
      "results": [
        "Similarity search in vector space relies on distance metrics to quantify how close two vectors are to each other. The three most common metrics are cosine similarity, Euclidean distance, and dot product. Cosine similarity measures the angle between vectors, making it insensitive to magnitude differences and ideal for comparing text embeddings where the direction of the vector carries the semantic",
        "the direction of the vector carries the semantic meaning. Euclidean distance measures the straight-line distance in the embedding space and works well when vector magnitudes are meaningful. Dot product combines both angle and magnitude, making it suitable for scenarios where vector length correlates with information density or confidence scores.",
        "The embedding dimensionality presents a practical tradeoff: higher-dimensional vectors (768 or 1024 dimensions) capture more semantic nuance but require more memory and slower comparison operations, while lower-dimensional vectors (384 dimensions) are faster to search and cheaper to store but may miss subtle semantic distinctions in complex queries."
      ]
    },
    "strategy_b": {
      "query_used": "User query: 'best vector database for semantic search'",
      "results": [
        "Hybrid search combines the strengths of vector similarity search with traditional keyword-based retrieval using techniques like reciprocal rank fusion (RRF). While semantic search excels at understanding conceptual similarity, it can miss exact keyword matches for proper nouns, product codes, or rare terms that appear infrequently in training data. A hybrid system runs both a vector search and a",
        "A hybrid system runs both a vector search and a lexical search (such as BM25) in parallel, then merges the result sets using RRF to produce a final ranking that balances semantic relevance with keyword precision. This approach consistently outperforms either method alone in production information retrieval benchmarks.",
        "Neighbors), developed by Google Research, use space-partitioning trees, quantization, and re-ranking stages to dramatically reduce the number of distance computations while maintaining high recall rates above 95%. Google's Vertex AI Vector Search is built on ScaNN technology and delivers low-latency similarity search for enterprise-scale deployments."
      ]
    }
  }
]
```

## Analysis

See `docs/similarity_metrics.md` for an explanation of the distance metric choice and `docs/vertex_ai_migration.md` for the production migration plan.
