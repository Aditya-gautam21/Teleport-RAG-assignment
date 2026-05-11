import os
from dotenv import load_dotenv
from llama_cpp import Llama

from services.vectorstore  import load_vectorstore
from utils.prompts import Prompts

load_dotenv()

def llm_expanded_search(query: str):
    llm = Llama(
        model_path=os.getenv('MODEL_PATH'),
        n_gpu_layers=8,
        verbose=False
    )

    prompt = Prompts.expanded_search()

    response = llm.create_chat_completion(
        messages=[
            {
                'role':'system',
                'content': prompt
            },
            {
                'role':'user',
                'content': query
            }
        ],
        temperature=0
    )

    return(response['choices'][0]['message']['content'])


def expanded_search(query: str, k: int = 3):
    expanded_query = llm_expanded_search(query)

    vs = load_vectorstore()
   
    results = vs.similarity_search(expanded_query, k)

    for i, doc in enumerate(results):
        print(f"----Result {i+1}----")
        print(doc.page_content)
        print()

    return results

if __name__ == '__main__':
    benchmark_query = "How can external knowledge bases prevent language models from making up false information?"
    expanded_search(benchmark_query)