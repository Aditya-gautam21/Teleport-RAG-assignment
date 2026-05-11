from langchain_huggingface import HuggingFaceEmbeddings
from services.chunker import text_splitter

def get_embeddings():
    chunks = text_splitter()
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    query = "How does the system handle peak load?"
    query_vector = embeddings.embed_query(query)

    return chunks, embeddings

if __name__ == '__main__':
    chunks, embeddings_model = get_embeddings()