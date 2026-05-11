from unittest.mock import MagicMock, patch

import pytest

from mocks.vertex_mocks import (
    MockTextEmbeddingModel,
    MockGenerativeModel,
    _TextEmbedding,
    _Prediction,
)


class TestMockTextEmbeddingModel:
    def test_model_name_default(self):
        model = MockTextEmbeddingModel()
        assert model.model_name == "textembedding-gecko"

    def test_model_name_custom(self):
        model = MockTextEmbeddingModel(model_name="custom-model")
        assert model.model_name == "custom-model"

    def test_get_embeddings_returns_list_of_textembedding(self):
        model = MockTextEmbeddingModel()
        result = model.get_embeddings(["hello world"])
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], _TextEmbedding)

    def test_get_embeddings_values_are_floats(self):
        model = MockTextEmbeddingModel()
        result = model.get_embeddings(["test"])
        assert isinstance(result[0].values, list)
        assert all(isinstance(v, float) for v in result[0].values)

    def test_get_embeddings_multiple_texts(self):
        model = MockTextEmbeddingModel()
        result = model.get_embeddings(["first", "second", "third"])
        assert len(result) == 3

    def test_get_embeddings_dimension_384(self):
        model = MockTextEmbeddingModel()
        result = model.get_embeddings(["test"])
        assert len(result[0].values) == 384


class TestMockGenerativeModel:
    def test_model_name_default(self):
        model = MockGenerativeModel()
        assert model.model_name == "gemini-pro"

    def test_model_name_custom(self):
        model = MockGenerativeModel(model_name="gemini-ultra")
        assert model.model_name == "gemini-ultra"

    def test_predict_returns_prediction_object(self, monkeypatch):
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "expanded query text"}}]
        }
        monkeypatch.setattr(
            "mocks.vertex_mocks.Llama", lambda **kwargs: mock_llm
        )

        model = MockGenerativeModel()
        result = model.predict("test prompt")

        assert isinstance(result, _Prediction)
        assert result.text == "expanded query text"

    def test_predict_passes_temperature_and_max_tokens(self, monkeypatch):
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        monkeypatch.setattr(
            "mocks.vertex_mocks.Llama", lambda **kwargs: mock_llm
        )

        model = MockGenerativeModel()
        model.predict("prompt", temperature=0.5, max_output_tokens=128)

        call_kwargs = mock_llm.create_chat_completion.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 128
