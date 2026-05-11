import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from services.embeddings import get_embeddings

chunks, embeddings = get_embeddings()

INDEX_PATH = Path('/home/adityagautam/Desktop/Projects/Teleport-RAG-assignment/data/vectorstore')
INDEX_FILE = INDEX_PATH / "index.faiss"

def create_vecotrstore():
    vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local(INDEX_PATH)
    return vector_store

def load_vectorstore():
    if not INDEX_FILE.exists():
        vector_store = create_vecotrstore()

    else:
        vector_store = FAISS.load_local(
            folder_path=INDEX_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
            )

    return vector_store