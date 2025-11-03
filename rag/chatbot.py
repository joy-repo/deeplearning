from transformers import pipeline
from langchain_chroma import Chroma
import gradio as gr
from typing import List


# Local SBERT wrapper (returns lists of floats compatible with Chroma)
class LLMStudioEmbeddingsWrapper:
    """Wrapper to call a local/remote LLM-Studio embeddings endpoint.

    Uses env vars:
      - LLM_STUDIO_EMBED_URL (default: http://127.0.0.1:8080/embed)
      - LLM_STUDIO_API_KEY

    Implements embed_documents and embed_query to be compatible with Chroma.
    """

    def __init__(self, endpoint: str | None = None, api_key: str | None = None):
        import os, requests
        self.endpoint = endpoint or os.getenv("LLM_STUDIO_EMBED_URL", "http://127.0.0.1:1234/v1/embeddings")
        self.api_key = api_key or os.getenv("LLM_STUDIO_API_KEY")
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        self.session.headers.update({"Content-Type": "application/json"})

    def _to_texts(self, texts: List[str]):
        processed = []
        for t in texts:
            if isinstance(t, str):
                processed.append(t)
            else:
                processed.append(getattr(t, "page_content", str(t)))
        return processed

    def embed_documents(self, texts: List[str]):
        processed = self._to_texts(texts)
        # Use OpenAI-compatible format
        payload = {
            "input": processed,
            "model": "embedding-model"  # Required for OpenAI compatibility
        }
        resp = self.session.post(self.endpoint, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        # OpenAI format response: {"data": [{"embedding": [...]}, ...]}
        if isinstance(data, dict) and "data" in data:
            return [item.get("embedding") for item in data["data"]]
        
        # Fallback for other formats
        if isinstance(data, dict) and "embeddings" in data:
            return data["embeddings"]
        if isinstance(data, list):
            return data
        
        raise ValueError("Unexpected response from embedding endpoint: %r" % data)

    def embed_query(self, text: str):
        return self.embed_documents([text])[0]

# import the .env file
from dotenv import load_dotenv
load_dotenv()

# configuration
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# Get vector store with embeddings configured
from ingest_database import get_vector_store

# Setup LM Studio client for chat completions
import requests
import json

def call_llm_studio(prompt):
    url = "http://127.0.0.1:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided knowledge. Only use the information in the knowledge section to answer questions. If the knowledge doesn't contain relevant information, say so."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 500,
        "stream": False
    }
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()

# connect to the chromadb using our configured vector store
vector_store = get_vector_store()

# Set up the vectorstore to be the retriever
num_results = 5
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

# call this function for every message added to the chatbot
def stream_response(message, history):
    # retrieve the relevant chunks based on the question asked
    docs = retriever.invoke(message)

    # add all the chunks to 'knowledge'
    knowledge = ""
    for doc in docs:
        knowledge += doc.page_content + "\n\n"

    # Create the full prompt with the RAG context
    rag_prompt = f"""Use the following knowledge to answer the question. If the knowledge doesn't contain relevant information, say so clearly.

Question: {message}

Knowledge:
{knowledge}

Answer the question based only on the provided knowledge."""

    # make the call to LM Studio
    if message is not None:
        try:
            response = call_llm_studio(rag_prompt)
            # Extract just the content from the LM Studio response
            content = response['choices'][0]['message']['content']
            # Stream the content in chunks
            partial_message = ""
            for i in range(0, len(content), 200):
                partial_message += content[i:i+200]
                yield partial_message
        except Exception as e:
            yield f"Error: Failed to get response from LM Studio. Make sure the server is running. Error: {str(e)}"

# initiate the Gradio app
chatbot = gr.ChatInterface(stream_response, textbox=gr.Textbox(placeholder="Send to the LLM...",
    container=False,
    autoscroll=True,
    scale=7),
)

# launch the Gradio app
chatbot.launch()