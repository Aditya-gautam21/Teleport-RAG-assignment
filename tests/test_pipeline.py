import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from services.pipeline import RAGPipeline


@pytest.fixture
def sample_doc():
    return [
        MagicMock(page_content="Chunk about vector search and FAISS indexing."),
        MagicMock(page_content="Chunk about RAG pipelines and LLM grounding."),
        MagicMock(page_content="Chunk about distance metrics like cosine similarity."),
    ]


@pytest.fixture
def pipeline_with_mocks(sample_doc):
    with patch("services.pipeline.MockTextEmbeddingModel") as mock_emb, \
         patch("services.pipeline.MockGenerativeModel") as mock_gen, \
         patch("services.pipeline.FAISS") as mock_faiss, \
         patch("services.pipeline.TextLoader") as mock_loader, \
         patch("services.pipeline.RecursiveCharacterTextSplitter") as mock_splitter:

        mock_emb_instance = mock_emb.return_value
        mock_emb_instance.get_embeddings.return_value = [
            MagicMock(values=[0.1, 0.2, 0.3])
        ]
        mock_emb_instance._model = MagicMock()

        mock_gen_instance = mock_gen.return_value
        mock_gen_instance.predict.return_value = MagicMock(
            text="expanded query keywords"
        )

        mock_splitter_instance = mock_splitter.return_value
        mock_splitter_instance.split_documents.return_value = sample_doc

        mock_loader_instance = mock_loader.return_value
        mock_loader_instance.load.return_value = sample_doc

        mock_faiss_instance = mock_faiss.from_documents.return_value
        mock_faiss.load_local.return_value = mock_faiss_instance
        mock_faiss_instance.similarity_search_by_vector.return_value = sample_doc[:2]

        pipeline = RAGPipeline()
        yield pipeline


class TestRAGPipelineInit:
    def test_default_index_path(self):
        pipeline = RAGPipeline()
        assert pipeline.index_path.name == "vectorstore"
        assert pipeline.index_file.name == "index.faiss"

    def test_custom_index_path(self, tmp_path):
        pipeline = RAGPipeline(index_path=tmp_path / "custom")
        assert pipeline.index_path == tmp_path / "custom"

    def test_vector_store_starts_none(self):
        pipeline = RAGPipeline()
        assert pipeline.vector_store is None

    def test_creates_embedder_with_gecko_model(self):
        pipeline = RAGPipeline()
        assert pipeline.embedder.model_name == "textembedding-gecko"

    def test_creates_expander_with_gemini_model(self):
        pipeline = RAGPipeline()
        assert pipeline.expander.model_name == "gemini-pro"


class TestRAGPipelineIngest:
    def test_ingest_creates_vector_store(self, pipeline_with_mocks):
        result = pipeline_with_mocks.ingest("dummy_path.txt")
        assert result is not None

    def test_ingest_calls_save_local(self, pipeline_with_mocks):
        pipeline_with_mocks.ingest("dummy_path.txt")
        pipeline_with_mocks.vector_store.save_local.assert_called_once()

    def test_ingest_returns_faiss_instance(self, pipeline_with_mocks):
        vs = pipeline_with_mocks.ingest("dummy_path.txt")
        assert vs == pipeline_with_mocks.vector_store


class TestRAGPipelineSearchRaw:
    def test_search_raw_returns_list(self, pipeline_with_mocks):
        pipeline_with_mocks.vector_store = MagicMock()
        pipeline_with_mocks.vector_store.similarity_search_by_vector.return_value = []
        pipeline_with_mocks.embedder.get_embeddings.return_value = [
            MagicMock(values=[0.1, 0.2])
        ]

        results = pipeline_with_mocks.search_raw("test query", k=3)
        assert isinstance(results, list)

    def test_search_raw_calls_get_embeddings(self, pipeline_with_mocks):
        pipeline_with_mocks.vector_store = MagicMock()
        pipeline_with_mocks.vector_store.similarity_search_by_vector.return_value = []
        pipeline_with_mocks.embedder.get_embeddings.return_value = [
            MagicMock(values=[0.1, 0.2])
        ]

        pipeline_with_mocks.search_raw("test query")
        pipeline_with_mocks.embedder.get_embeddings.assert_called_once_with(["test query"])

    def test_search_raw_uses_similarity_search_by_vector(self, pipeline_with_mocks):
        pipeline_with_mocks.vector_store = MagicMock()
        pipeline_with_mocks.vector_store.similarity_search_by_vector.return_value = []
        pipeline_with_mocks.embedder.get_embeddings.return_value = [
            MagicMock(values=[0.1, 0.2])
        ]

        pipeline_with_mocks.search_raw("query", k=5)
        call_args = pipeline_with_mocks.vector_store.similarity_search_by_vector.call_args
        assert call_args.kwargs["k"] == 5


class TestRAGPipelineSearchEnhanced:
    def test_search_enhanced_returns_expanded_query_and_results(self, pipeline_with_mocks):
        pipeline_with_mocks.vector_store = MagicMock()
        pipeline_with_mocks.vector_store.similarity_search_by_vector.return_value = []
        pipeline_with_mocks.embedder.get_embeddings.return_value = [
            MagicMock(values=[0.1, 0.2])
        ]

        expanded, results = pipeline_with_mocks.search_enhanced("test query")
        assert expanded == "expanded query keywords"
        assert isinstance(results, list)

    def test_search_enhanced_calls_predict(self, pipeline_with_mocks):
        pipeline_with_mocks.vector_store = MagicMock()
        pipeline_with_mocks.vector_store.similarity_search_by_vector.return_value = []
        pipeline_with_mocks.embedder.get_embeddings.return_value = [
            MagicMock(values=[0.1, 0.2])
        ]

        pipeline_with_mocks.search_enhanced("test query")
        pipeline_with_mocks.expander.predict.assert_called_once()

    def test_search_enhanced_embeds_expanded_query(self, pipeline_with_mocks):
        pipeline_with_mocks.vector_store = MagicMock()
        pipeline_with_mocks.vector_store.similarity_search_by_vector.return_value = []
        pipeline_with_mocks.embedder.get_embeddings.return_value = [
            MagicMock(values=[0.1, 0.2])
        ]

        pipeline_with_mocks.search_enhanced("test query")
        pipeline_with_mocks.embedder.get_embeddings.assert_called_once_with(
            ["expanded query keywords"]
        )


class TestRAGPipelineBenchmark:
    def test_benchmark_returns_list_of_dicts(self, pipeline_with_mocks):
        pipeline_with_mocks.vector_store = MagicMock()
        pipeline_with_mocks.vector_store.similarity_search_by_vector.return_value = []
        pipeline_with_mocks.embedder.get_embeddings.return_value = [
            MagicMock(values=[0.1, 0.2])
        ]

        results = pipeline_with_mocks.benchmark(["q1", "q2", "q3"], k=2)
        assert len(results) == 3

    def test_benchmark_entry_structure(self, pipeline_with_mocks):
        pipeline_with_mocks.vector_store = MagicMock()
        pipeline_with_mocks.vector_store.similarity_search_by_vector.return_value = []
        pipeline_with_mocks.embedder.get_embeddings.return_value = [
            MagicMock(values=[0.1, 0.2])
        ]

        results = pipeline_with_mocks.benchmark(["sample query"], k=2)
        entry = results[0]
        assert entry["query"] == "sample query"
        assert "strategy_a" in entry
        assert "strategy_b" in entry
        assert "query_used" in entry["strategy_a"]
        assert "query_used" in entry["strategy_b"]
        assert "results" in entry["strategy_a"]
        assert "results" in entry["strategy_b"]

    def test_benchmark_strategy_a_uses_raw_query(self, pipeline_with_mocks):
        pipeline_with_mocks.vector_store = MagicMock()
        pipeline_with_mocks.vector_store.similarity_search_by_vector.return_value = []
        pipeline_with_mocks.embedder.get_embeddings.return_value = [
            MagicMock(values=[0.1, 0.2])
        ]

        results = pipeline_with_mocks.benchmark(["raw input query"], k=2)
        assert results[0]["strategy_a"]["query_used"] == "raw input query"

    def test_benchmark_strategy_b_uses_expanded_query(self, pipeline_with_mocks):
        pipeline_with_mocks.vector_store = MagicMock()
        pipeline_with_mocks.vector_store.similarity_search_by_vector.return_value = []
        pipeline_with_mocks.embedder.get_embeddings.return_value = [
            MagicMock(values=[0.1, 0.2])
        ]

        results = pipeline_with_mocks.benchmark(["raw input query"], k=2)
        assert results[0]["strategy_b"]["query_used"] == "expanded query keywords"

    def test_benchmark_to_json_returns_valid_json(self, pipeline_with_mocks):
        pipeline_with_mocks.vector_store = MagicMock()
        pipeline_with_mocks.vector_store.similarity_search_by_vector.return_value = []
        pipeline_with_mocks.embedder.get_embeddings.return_value = [
            MagicMock(values=[0.1, 0.2])
        ]

        json_str = pipeline_with_mocks.benchmark_to_json(["q1"], k=2)
        parsed = json.loads(json_str)
        assert len(parsed) == 1
