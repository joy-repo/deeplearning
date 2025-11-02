from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from uuid import uuid4
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import os
import re
import time
import random
import logging
from requests.auth import HTTPBasicAuth

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

class LLMStudioEmbeddingsWrapper:
    """Wrapper that calls an LLM‑Studio (or similar) HTTP embedding endpoint.

    Expects an endpoint that accepts POST {"input": [..texts..], "model": "embedding-model"}
    and returns JSON in OpenAI format: {"data": [{"embedding": [...]}, ...]}
    
    The endpoint and API key are configurable via environment variables:
      - LLM_STUDIO_EMBED_URL (default: http://127.0.0.1:1234/v1/embeddings)
      - LLM_STUDIO_API_KEY

    This object implements embed_documents and embed_query to be compatible
    with Chroma/langchain embedding interfaces.
    """

    def __init__(self, endpoint: str | None = None, api_key: str | None = None):
        self.endpoint = endpoint or os.getenv("LLM_STUDIO_EMBED_URL", "http://127.0.0.1:1234/v1/embeddings")
        self.api_key = api_key or os.getenv("LLM_STUDIO_API_KEY")
        self.session = requests.Session()
        # attach API key if present
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        self.session.headers.update({"Content-Type": "application/json"})

    def _to_texts(self, texts):
        processed = []
        for t in texts:
            if isinstance(t, str):
                processed.append(t)
            else:
                processed.append(getattr(t, "page_content", str(t)))
        return processed

    def embed_documents(self, texts):
        processed = self._to_texts(texts)
        # Use OpenAI-compatible format
        payload = {
            "input": processed,
            "model": "embedding-model"  # This is required for OpenAI compatibility
        }
        logging.info(f"Sending embedding request to {self.endpoint} with {len(processed)} texts")
        resp = self.session.post(self.endpoint, json=payload, timeout=30)
        logging.info(f"LM Studio response: {resp.status_code} - {resp.reason}")
        
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
        result = self.embed_documents([text])
        return result[0]


def get_embeddings_model():
    return LLMStudioEmbeddingsWrapper()

def get_vector_store():
    embeddings_model = get_embeddings_model()
    return Chroma(
        collection_name="llm_studio_collection",  # Different name for 768-dim embeddings
        embedding_function=embeddings_model,
        persist_directory=CHROMA_PATH,
    )

def fetch_url_text(url: str) -> str:
    """Fetch the main text of `url` and return as plain text.

    Uses a Session with a browser-like User-Agent and a Retry policy to
    reduce 403/429/temporary network errors when scraping common sites.
    """
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504))
    session.mount("https://", HTTPAdapter(max_retries=retries))

    confl_pattern = re.compile(r"https?://([^.]+\.atlassian\.net)/.*/pages/(\d+)")
    m = confl_pattern.search(url)
    if m:
        base_host = m.group(1)
        page_id = m.group(2)

        base_url = os.getenv("CONFLUENCE_BASE_URL", f"https://{base_host}")
        email = os.getenv("CONFLUENCE_EMAIL")
        token = os.getenv("CONFLUENCE_API_TOKEN")

        if email and token:
            api_url = f"{base_url}/wiki/rest/api/content/{page_id}?expand=body.view"
            resp = session.get(api_url, auth=HTTPBasicAuth(email, token), headers={"Accept": "application/json"}, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            html = data.get("body", {}).get("view", {}).get("value", "")
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "header", "footer", "nav", "noscript"]):
                tag.decompose()
            return "\n".join(soup.stripped_strings)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    resp = session.get(url, headers=headers, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav", "noscript"]):
        tag.decompose()
    return "\n".join(soup.stripped_strings)


urls_file = Path(DATA_PATH) / "urls.txt"
if urls_file.exists():
    with open(urls_file, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]
else:
    urls = [
        "https://example.com",
    ]


class SimpleDoc:
    def __init__(self, page_content: str, metadata: dict | None = None):
        self.page_content = page_content
        self.metadata = metadata or {}


raw_documents = []
for url in urls:
    try:
        text = fetch_url_text(url)
        raw_documents.append(SimpleDoc(text, metadata={"source": url}))
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)


chunks = text_splitter.split_documents(raw_documents)


uuids = [str(uuid4()) for _ in range(len(chunks))]


def chunked(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i+size]


def add_documents_with_retries(vector_store, docs, ids, batch_size=64, max_retries=5):
    for doc_batch, id_batch in zip(chunked(docs, batch_size), chunked(ids, batch_size)):
        attempt = 0
        while True:
            try:
                vector_store.add_documents(documents=doc_batch, ids=id_batch)
                break
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    print(f"Failed after {max_retries} attempts; last error: {e}")
                    raise
                backoff = (5 ** attempt) + random.random()
                print(f"Transient error, retrying in {backoff:.1f}s (attempt {attempt}/{max_retries}): {e}")
                time.sleep(backoff)


if __name__ == "__main__":
    vector_store = get_vector_store()
    add_documents_with_retries(vector_store, chunks, uuids, batch_size=64, max_retries=5)
    print("completed ingesting documents.")