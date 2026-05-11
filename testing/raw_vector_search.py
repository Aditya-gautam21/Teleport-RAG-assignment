from services.vectorstore  import load_vectorstore

def raw_search(query: str, k: int = 3):
    vs = load_vectorstore()

    results = vs.similarity_search(query, k)

    for i, doc in enumerate(results):
        print(f"----Result {i+1}----")
        print(doc.page_content)
        print()

    return results

if __name__ == '__main__':
    benchmark_query = "How does the system handle peak load?"
    raw_search(benchmark_query)