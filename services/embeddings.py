import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from services.chunker import text_splitter

def get_embeddings():
    chunks = text_splitter()
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    return chunks, embeddings