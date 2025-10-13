from transformers import pipeline
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
import gradio as gr
from typing import List


# Local SBERT wrapper (returns lists of floats compatible with Chroma)
class SBERTEmbeddingsWrapper:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

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
        embeddings = self.model.encode(processed, show_progress_bar=False)
        return [emb.tolist() if hasattr(emb, "tolist") else emb for emb in embeddings]

    def embed_query(self, text: str):
        emb = self.model.encode([text])
        return emb[0].tolist()

# import the .env file
from dotenv import load_dotenv
load_dotenv()

# configuration
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

embeddings_model = SBERTEmbeddingsWrapper()

# initiate a local text-generation / text2text model (no OpenAI)
# Using Flan-T5 small as a lightweight instruction-following model; change
# to a different local HF model if you prefer.
llm = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)

# connect to the chromadb
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH, 
)

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

        # generate the response locally and yield it in small chunks to
        # approximate streaming behavior
        full = llm(rag_prompt, max_length=512)[0]["generated_text"]
        # yield in 200-char chunks
        for i in range(0, len(full), 200):
            partial_message += full[i:i+200]
            yield partial_message

# initiate the Gradio app
chatbot = gr.ChatInterface(stream_response, textbox=gr.Textbox(placeholder="Send to the LLM...",
    container=False,
    autoscroll=True,
    scale=7),
)

# launch the Gradio app
chatbot.launch()