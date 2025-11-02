from langchain_chroma import Chroma
import gradio as gr
from typing import List
import os
import requests


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
        print(f"DEBUG: Sending request to {self.endpoint}")
        print(f"DEBUG: Payload: {payload}")
        resp = self.session.post(self.endpoint, json=payload, timeout=30)
        print(f"DEBUG: Response status: {resp.status_code}")
        print(f"DEBUG: Response text: {resp.text}")
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

# LM Studio chat completion wrapper
class LMStudioChatWrapper:
    def __init__(self, endpoint: str | None = None, api_key: str | None = None):
        self.endpoint = endpoint or os.getenv("LLM_STUDIO_CHAT_URL", "http://127.0.0.1:1234/v1/chat/completions")
        self.api_key = api_key or os.getenv("LLM_STUDIO_API_KEY")
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        self.session.headers.update({"Content-Type": "application/json"})
    
    def generate(self, prompt: str, max_tokens: int = 512):
        payload = {
            "model": "local-model",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": False
        }
        
        try:
            resp = self.session.post(self.endpoint, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"LM Studio chat error: {e}")
            # Fallback to a simple response
            return f"I apologize, but I'm having trouble connecting to the chat service. Error: {str(e)}"

# Initialize LM Studio chat wrapper
llm = LMStudioChatWrapper()

# connect to the chromadb using our configured vector store
vector_store = get_vector_store()

# Set up the vectorstore to be the retriever
num_results = 5
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

# call this function for every message added to the chatbot
def stream_response(message, history):
    #print(f"Input: {message}. History: {history}\n")

    # retrieve the relevant chunks based on the question asked
    docs = retriever.invoke(message)

    # add all the chunks to 'knowledge'
    knowledge = ""

    for doc in docs:
        knowledge += doc.page_content+"\n\n"


    # make the call to the LLM (including prompt)
    if message is not None:
        partial_message = ""

        rag_prompt = f"""
You are an assistant which answers questions based on knowledge which is provided to you.
While answering, you don't use your internal knowledge,
but solely the information in the "The knowledge" section.
You don't mention anything to the user about the provided knowledge.

The question: {message}

Conversation history: {history}

The knowledge: {knowledge}

"""

        # print(rag_prompt)

        try:
            # generate the response using LM Studio chat completions
            full = llm.generate(rag_prompt, max_tokens=512)
            # yield in 200-char chunks to approximate streaming behavior
            for i in range(0, len(full), 200):
                partial_message += full[i:i+200]
                yield partial_message
        except Exception as e:
            yield f"Error generating response: {str(e)}"

# initiate the Gradio app
chatbot = gr.ChatInterface(stream_response, textbox=gr.Textbox(placeholder="Send to the LLM...",
    container=False,
    autoscroll=True,
    scale=7),
)

# launch the Gradio app
chatbot.launch()