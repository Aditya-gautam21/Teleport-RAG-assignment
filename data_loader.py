from pathlib import Path
from langchain_community.document_loaders import TextLoader

def load_data():
    file_path = Path('/home/adityagautam/Desktop/Projects/Teleport-RAG-assignment/sample.txt')

    loader = TextLoader(file_path=file_path)
    text = loader.load()

    return text