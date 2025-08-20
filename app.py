import os
import re
import time
import json
import math
import queue
import hashlib
import requests
import streamlit as st
import numpy as np
from typing import List, Dict, Any, Tuple, Set
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

try:
    import torch
except ImportError:
    torch = None

try:
    from InstructorEmbedding import INSTRUCTOR
except ImportError:
    INSTRUCTOR = None

from openai import OpenAI


# ---------- Config ----------
APP_TITLE = "RAG Chat over prabhupadabooks.com"
DEFAULT_BASE_URL = "https://prabhupadabooks.com"
DEFAULT_MAX_PAGES = 150
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
DOC_EMBED_INSTRUCTION = "Represent the document for retrieval:"
QUERY_EMBED_INSTRUCTION = "Represent the question for retrieving supporting documents:"
STORE_ROOT = ".rag_store"


# ---------- Utilities ----------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def get_domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def same_domain(url: str, base_domain: str) -> bool:
    return get_domain(url) == base_domain


def is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    if any(url.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".pdf", ".zip", ".mp3", ".mp4", ".avi", ".mov", ".wmv", ".mkv", ".webm", ".ico", ".css", ".js"]):
        return False
    return True


def hash_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()[:16]


def get_store_dir(base_url: str, embed_model_name: str) -> str:
    host = get_domain(base_url)
    safe_model = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", embed_model_name)
    return os.path.join(STORE_ROOT, f"{host}__{safe_model}")


# ---------- Scraping ----------
def fetch_html(url: str, timeout: int = 20) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0; +https://example.com/bot)"}
    r = requests.get(url, timeout=timeout, headers=headers)
    r.raise_for_status()
    return r.text


def extract_clean_text(html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form", "iframe"]):
        tag.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    text = soup.get_text(separator=" ", strip=True)
    text = normalize_whitespace(text)
    return title, text


def discover_links(html: str, base_url: str) -> Set[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href:
            continue
        abs_url = urljoin(base_url, href)
        links.add(abs_url)
    return links


def crawl_site(base_url: str, max_pages: int, delay: float = 0.5) -> List[Dict[str, Any]]:
    visited: Set[str] = set()
    to_visit = queue.Queue()
    to_visit.put(base_url)
    base_domain = get_domain(base_url)
    pages: List[Dict[str, Any]] = []
    count = 0

    while not to_visit.empty() and count < max_pages:
        url = to_visit.get()
        if url in visited or not is_valid_url(url) or not same_domain(url, base_domain):
            continue
        visited.add(url)
        try:
            html = fetch_html(url)
        except Exception:
            continue

        title, text = extract_clean_text(html)
        if len(text) > 0:
            pages.append({"url": url, "title": title, "text": text})
            count += 1

        for link in discover_links(html, url):
            if link not in visited and is_valid_url(link) and same_domain(link, base_domain):
                to_visit.put(link)

        time.sleep(delay)
    return pages


# ---------- Chunking ----------
def chunk_text(text: str, max_chars: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


# ---------- Embedding ----------
class InstructorEmbedder:
    def __init__(self, model_name: str):
        if INSTRUCTOR is None:
            raise RuntimeError("InstructorEmbedding is not installed. Please install with: pip install InstructorEmbedding")
        device = "cpu"
        if torch is not None and torch.cuda.is_available():
            device = "cuda"
        self.model_name = model_name
        self.device = device
        self.model = INSTRUCTOR(model_name)
        if hasattr(self.model, "to"):
            try:
                self.model.to(device)
            except Exception:
                pass

    def embed_texts(self, texts: List[str], instruction: str, batch_size: int = 8, show_progress_bar: bool = True) -> np.ndarray:
        inputs = [[instruction, t] for t in texts]
        vecs = self.model.encode(
            inputs,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True
        )
        return np.array(vecs, dtype=np.float32)


@st.cache_resource(show_spinner=True)
def load_embedder(model_name: str) -> InstructorEmbedder:
    return InstructorEmbedder(model_name)


# ---------- Indexing ----------
def build_corpus(pages: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    for p in pages:
        url = p["url"]
        title = p.get("title", "")
        text = p["text"]
        chunks = chunk_text(text)
        for i, ch in enumerate(chunks):
            texts.append(ch)
            metas.append({
                "url": url,
                "title": title,
                "chunk_id": i,
                "source_id": hash_text(url + str(i))
            })
    return texts, metas


def compute_embeddings(embedder: InstructorEmbedder, texts: List[str], instruction: str, batch_size: int = 8) -> np.ndarray:
    return embedder.embed_texts(texts, instruction=instruction, batch_size=batch_size, show_progress_bar=True)


def save_index(store_dir: str, embeddings: np.ndarray, texts: List[str], metas: List[Dict[str, Any]], config: Dict[str, Any]):
    ensure_dir(store_dir)
    np.save(os.path.join(store_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(store_dir, "texts.jsonl"), "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    with open(os.path.join(store_dir, "metas.jsonl"), "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    with open(os.path.join(store_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def load_index(store_dir: str) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]], Dict[str, Any]]:
    emb_path = os.path.join(store_dir, "embeddings.npy")
    txt_path = os.path.join(store_dir, "texts.jsonl")
    meta_path = os.path.join(store_dir, "metas.jsonl")
    cfg_path = os.path.join(store_dir, "config.json")
    if not (os.path.exists(emb_path) and os.path.exists(txt_path) and os.path.exists(meta_path) and os.path.exists(cfg_path)):
        raise FileNotFoundError("Index files not found. Please build the index first.")
    embeddings = np.load(emb_path)
    texts = []
    metas = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return embeddings, texts, metas, config


# ---------- Retrieval ----------
def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # assumes embeddings are normalized
    return np.dot(a, b.T)


def retrieve(query_vec: np.ndarray, doc_embeddings: np.ndarray, top_k: int = 5) -> List[int]:
    sims = cosine_similarity_matrix(query_vec.reshape(1, -1), doc_embeddings)  # shape (1, N)
    sims = sims.flatten()
    top_idx = np.argsort(-sims)[:top_k]
    return top_idx.tolist()


# ---------- OpenAI Client ----------
@st.cache_resource(show_spinner=False)
def get_openai_client() -> OpenAI:
    return OpenAI()


def generate_answer(client: OpenAI, model: str, system_prompt: str, messages: List[Dict[str, str]]) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}] + messages
    )
    return response.choices[0].message.content


# ---------- RAG Pipeline ----------
def build_context(selected_idxs: List[int], texts: List[str], metas: List[Dict[str, Any]], max_chars: int = 6000) -> Tuple[str, List[Dict[str, Any]]]:
    blocks = []
    sources = []
    total = 0
    for idx in selected_idxs:
        meta = metas[idx]
        url = meta["url"]
        title = meta.get("title", "")
        chunk = texts[idx]
        snippet = chunk[:1200]
        block = f"[Title] {title}\n[URL] {url}\n[Content]\n{snippet}"
        size = len(block)
        if total + size > max_chars and blocks:
            break
        total += size
        blocks.append(block)
        sources.append({"url": url, "title": title})
    context = "\n\n---\n\n".join(blocks)
    return context, sources


def format_messages(chat_history: List[Dict[str, str]], user_query: str, context: str) -> List[Dict[str, str]]:
    preface = (
        "You are a RAG assistant answering strictly from the provided CONTEXT which was scraped from https://prabhupadabooks.com. "
        "Follow rules:\n"
        "- Use only the CONTEXT to answer. If the answer is not present, say you don't know based on the available sources.\n"
        "- Provide concise, accurate answers.\n"
        "- Always include a References section listing the exact source URLs used.\n"
        "- Add this clause verbatim at the end: 'Answer based on text scraped from https://prabhupadabooks.com.'"
    )
    ctx_block = f"CONTEXT START\n{context}\nCONTEXT END"
    messages = []
    for m in chat_history[-6:]:
        messages.append(m)
    messages.append({"role": "user", "content": f"{preface}\n\n{ctx_block}\n\nQuestion: {user_query}"})
    return messages


# ---------- Streamlit App ----------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    with st.sidebar:
        st.header("Configuration")

        api_key = st.text_input("OpenAI API Key", type="password", help="Set your OpenAI API key to enable GPT-4 responses.")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        base_url = st.text_input("Base URL to crawl", value=DEFAULT_BASE_URL)
        max_pages = st.slider("Max pages to crawl", min_value=10, max_value=1000, value=DEFAULT_MAX_PAGES, step=10)
        embed_model_name = st.selectbox(
            "Embedding model",
            options=["hkunlp/instructor-xl", "hkunlp/instructor-large", "hkunlp/instructor-base"],
            index=0,
            help="Instructor-XL is very large and may require a GPU. Choose a smaller model if you face memory issues."
        )
        top_k = st.slider("Top K retrieved chunks", min_value=1, max_value=12, value=5)
        build_button = st.button("Crawl and Build Index", type="primary")
        load_button = st.button("Load Existing Index")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "index" not in st.session_state:
        st.session_state.index = None
    if "store_dir" not in st.session_state:
        st.session_state.store_dir = None

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Index Status")

        store_dir = get_store_dir(base_url, embed_model_name)
        st.session_state.store_dir = store_dir

        if build_button:
            with st.spinner("Loading embedder..."):
                try:
                    embedder = load_embedder(embed_model_name)
                except Exception as e:
                    st.error(f"Failed to load embedder '{embed_model_name}': {e}")
                    st.stop()

            st.info(f"Crawling up to {max_pages} pages from {base_url} ...")
            pages = crawl_site(base_url, max_pages=max_pages, delay=0.4)
            if not pages:
                st.warning("Crawl found no pages. Please adjust settings.")
                st.stop()

            st.write(f"Fetched {len(pages)} pages. Building corpus...")
            texts, metas = build_corpus(pages)
            st.write(f"Created {len(texts)} text chunks for embedding.")

            with st.spinner("Computing embeddings (this may take time)..."):
                try:
                    embeddings = compute_embeddings(embedder, texts, instruction=DOC_EMBED_INSTRUCTION, batch_size=4)
                except Exception as e:
                    st.error(f"Embedding failed: {e}")
                    st.stop()

            config = {
                "base_url": base_url,
                "embed_model_name": embed_model_name,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "doc_instruction": DOC_EMBED_INSTRUCTION,
                "query_instruction": QUERY_EMBED_INSTRUCTION
            }
            save_index(store_dir, embeddings, texts, metas, config)
            st.success(f"Index built and saved to {store_dir}")
            st.session_state.index = (embeddings, texts, metas, config)

        if load_button:
            try:
                data = load_index(store_dir)
                st.session_state.index = data
                st.success(f"Loaded index from {store_dir}")
            except Exception as e:
                st.error(f"Failed to load index: {e}")

        if st.session_state.index is not None:
            embeddings, texts, metas, config = st.session_state.index
            st.success(f"Index ready. Documents: {embeddings.shape[0]}, Dim: {embeddings.shape[1]}")

    with col2:
        st.subheader("Controls")
        clear = st.button("Clear Chat History")
        if clear:
            st.session_state.chat_history = []
            st.experimental_rerun()

        if st.session_state.index is None:
            st.info("Build or load an index first.")

    st.markdown("---")
    st.subheader("Chat")

    if st.session_state.index is None:
        st.info("Please build or load the index from the sidebar to start chatting.")
        return

    # Chat interface
    user_query = st.text_input("Ask a question based on content from prabhupadabooks.com")
    if st.button("Send", type="primary"):
        if not user_query.strip():
            st.warning("Please enter a question.")
            return

        if not os.getenv("OPENAI_API_KEY"):
            st.error("Please provide your OpenAI API Key in the sidebar.")
            return

        # Prepare retrieval
        embeddings, texts, metas, config = st.session_state.index
        try:
            embedder = load_embedder(config["embed_model_name"])
        except Exception as e:
            st.error(f"Failed to load embedder for querying: {e}")
            return

        with st.spinner("Retrieving relevant context..."):
            try:
                q_vec = compute_embeddings(embedder, [user_query], instruction=QUERY_EMBED_INSTRUCTION, batch_size=1)[0]
                idxs = retrieve(q_vec, embeddings, top_k=top_k)
                context, sources = build_context(idxs, texts, metas)
            except Exception as e:
                st.error(f"Retrieval failed: {e}")
                return

        # Prepare LLM messages
        messages = format_messages(st.session_state.chat_history, user_query, context)

        # Call OpenAI
        try:
            client = get_openai_client()
            system_prompt = "You are a helpful assistant."
            answer = generate_answer(client, model="gpt-4", system_prompt=system_prompt, messages=messages)
        except Exception as e:
            st.error(f"OpenAI completion failed: {e}")
            return

        # Update chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        # Display
        st.markdown("### Assistant")
        st.write(answer)

        st.markdown("### References")
        # Deduplicate sources while preserving order
        seen = set()
        unique_sources = []
        for s in sources:
            key = s["url"]
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)
        for s in unique_sources:
            st.write(f"- {s.get('title','(no title)')} — {s['url']}")

    # Render chat history
    if st.session_state.chat_history:
        st.markdown("### Chat History")
        for m in st.session_state.chat_history:
            if m["role"] == "user":
                st.markdown(f"User: {m['content']}")
            else:
                st.markdown(f"Assistant: {m['content']}")


if __name__ == "__main__":
    main()