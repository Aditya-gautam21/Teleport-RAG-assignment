from langchain_text_splitters import RecursiveCharacterTextSplitter
from services.data_loader import load_data

def text_splitter():
    text = load_data()
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    chunks = splitter.split_documents(text)
    return chunks