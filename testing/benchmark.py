import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.pipeline import RAGPipeline

BENCHMARK_QUERIES = [
    "How can external knowledge bases prevent language models from making up false information?",
    "What are the tradeoffs between exact and approximate nearest neighbor search for large-scale vector retrieval?",
    "How does chunk size and overlap affect the quality of retrieved documents in a RAG pipeline?",
    "When should a team choose cosine similarity over Euclidean distance for embedding-based search?",
]


def print_table(results):
    for i, entry in enumerate(results):
        print(f"{'='*80}")
        print(f"QUERY {i+1}: {entry['query']}")
        print(f"{'='*80}")

        print(f"\n--- Strategy A (Raw Vector Search) ---")
        for j, text in enumerate(entry["strategy_a"]["results"]):
            print(f"  [Rank {j+1}] {text[:200]}...")

        print(f"\n--- Strategy B (AI-Enhanced Search) ---")
        print(f"  Expanded query: {entry['strategy_b']['query_used'][:200]}...")
        for j, text in enumerate(entry["strategy_b"]["results"]):
            print(f"  [Rank {j+1}] {text[:200]}...")
        print()


def main():
    data_file = Path(__file__).resolve().parent.parent / "sample_data.txt"
    output_file = Path(__file__).resolve().parent.parent / "retrieval_benchmark.md"

    pipeline = RAGPipeline()
    pipeline.ingest(str(data_file))

    results = pipeline.benchmark(BENCHMARK_QUERIES, k=3)
    json_output = pipeline.benchmark_to_json(BENCHMARK_QUERIES, k=3)

    print_table(results)

    with open(output_file, "w") as f:
        f.write("# Retrieval Benchmark: Strategy A vs Strategy B\n\n")
        f.write("## Methodology\n\n")
        f.write("Strategy A embeds the raw user query directly and performs FAISS similarity search. ")
        f.write("Strategy B first expands the query using a local LLM (Gemma 4B) to add domain-specific ")
        f.write("keywords and alternative phrasings, then embeds the expanded query and searches the same index.\n\n")
        f.write("## Results\n\n")
        f.write("```json\n")
        f.write(json_output)
        f.write("\n```\n")
        f.write("\n## Analysis\n\n")
        f.write("See `docs/similarity_metrics.md` for an explanation of the distance metric choice ")
        f.write("and `docs/vertex_ai_migration.md` for the production migration plan.\n")

    print(f"Benchmark report written to {output_file}")


if __name__ == "__main__":
    main()
