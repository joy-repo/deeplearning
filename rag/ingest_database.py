from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
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
from requests.auth import HTTPBasicAuth

from dotenv import load_dotenv
load_dotenv()

DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

class SBERTEmbeddingsWrapper:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

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
        embeddings = self.model.encode(processed, show_progress_bar=False)
        return [emb.tolist() if hasattr(emb, "tolist") else emb for emb in embeddings]

    def embed_query(self, text: str):
        emb = self.model.encode([text])
        return emb[0].tolist()


embeddings_model = SBERTEmbeddingsWrapper()

vector_store = Chroma(
    collection_name="example_collection",
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
                backoff = (2 ** attempt) + random.random()
                print(f"Transient error, retrying in {backoff:.1f}s (attempt {attempt}/{max_retries}): {e}")
                time.sleep(backoff)


add_documents_with_retries(vector_store, chunks, uuids, batch_size=64, max_retries=5)
print("completed ingesting documents.")