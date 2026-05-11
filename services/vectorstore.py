import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from services.embeddings import get_embeddings
from services.chunker import text_splitter

INDEX_PATH = Path('/home/adityagautam/Desktop/Projects/Teleport-RAG-assignment/data/vectorstore')
INDEX_FILE = INDEX_PATH / "index.faiss"

def create_vectorstore():
    chunks = text_splitter()
    embeddings = get_embeddings()
    
    vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local(INDEX_PATH)
    return vector_store

def load_vectorstore():
    embeddings = get_embeddings()
    
    if not INDEX_FILE.exists():
        vector_store = create_vectorstore()

    else:
        vector_store = FAISS.load_local(
            folder_path=INDEX_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
            )

    return vector_store