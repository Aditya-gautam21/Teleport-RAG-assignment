import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from llama_cpp import Llama

load_dotenv()


class _TextEmbedding:
    """Replicates vertexai.language_models.TextEmbedding"""
    def __init__(self, values):
        self.values = values


class _Prediction:
    """Replicates vertexai.language_models.Prediction"""
    def __init__(self, text):
        self.text = text


class MockTextEmbeddingModel:
    """
    Mocks vertexai.language_models.TextEmbeddingModel.
    Uses local HuggingFace sentence-transformers under the hood.
    """
    def __init__(self, model_name="textembedding-gecko"):
        self.model_name = model_name
        self._model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def get_embeddings(self, texts):
        vectors = self._model.embed_documents(texts)
        return [_TextEmbedding(values=v) for v in vectors]


class MockGenerativeModel:
    """
    Mocks vertexai.language_models.GenerativeModel.
    Uses a local GGUF model via llama-cpp under the hood.
    """
    def __init__(self, model_name="gemini-pro"):
        self.model_name = model_name
        self._llm = Llama(
            model_path=os.getenv("MODEL_PATH"),
            n_gpu_layers=8,
            verbose=False,
        )

    def predict(self, prompt, temperature=0, max_output_tokens=256):
        response = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
        text = response["choices"][0]["message"]["content"]
        return _Prediction(text=text)
