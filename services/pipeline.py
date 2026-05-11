import json
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from mocks.vertex_mocks import MockTextEmbeddingModel, MockGenerativeModel
from utils.prompts import Prompts


class RAGPipeline:
    def __init__(self, index_path=None):
        self.embedder = MockTextEmbeddingModel(model_name="textembedding-gecko")
        self.expander = MockGenerativeModel(model_name="gemini-pro")
        self.vector_store = None

        if index_path is None:
            index_path = Path(__file__).resolve().parent.parent / "data" / "vectorstore"
        self.index_path = Path(index_path)
        self.index_file = self.index_path / "index.faiss"

    def ingest(self, file_path):
        loader = TextLoader(file_path=file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        self.vector_store = FAISS.from_documents(chunks, self.embedder._model)
        self.vector_store.save_local(str(self.index_path))
        return self.vector_store

    def load_index(self, file_path=None):
        if not self.index_file.exists():
            if file_path is None:
                raise ValueError("index not found. Provide file_path to ingest data.")
            self.vector_store = self.ingest(file_path)

        else:
            self.vector_store = FAISS.load_local(
                folder_path=str(self.index_path),
                embeddings=self.embedder._model,
                allow_dangerous_deserialization=True,
            )
        return self.vector_store

    def _ensure_index(self):
        if self.vector_store is None:
            self.load_index()

    def search_raw(self, query, k=3):
        self._ensure_index()
        query_vector = self.embedder.get_embeddings([query])[0].values
        return self.vector_store.similarity_search_by_vector(query_vector, k)

    def search_enhanced(self, query, k=3):
        self._ensure_index()
        prompt = Prompts.expanded_search()

        prediction = self.expander.predict(prompt, temperature=0)
        expanded_query = prediction.text.strip()

        query_vector = self.embedder.get_embeddings([expanded_query])[0].values
        return expanded_query, self.vector_store.similarity_search_by_vector(query_vector, k)

    def benchmark(self, queries, k=3):
        self._ensure_index()

        results = []
        for query in queries:
            raw_results = self.search_raw(query, k)
            expanded_query, enhanced_results = self.search_enhanced(query, k)

            results.append({
                "query": query,
                "strategy_a": {
                    "query_used": query,
                    "results": [doc.page_content for doc in raw_results],
                },
                "strategy_b": {
                    "query_used": expanded_query,
                    "results": [doc.page_content for doc in enhanced_results],
                },
            })

        return results

    def benchmark_to_json(self, queries, k=3):
        return json.dumps(self.benchmark(queries, k), indent=2)
