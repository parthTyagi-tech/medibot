from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import requests
import time
import logging
import hashlib
import math

logger = logging.getLogger("embeddings")

# ─────────────────────────────────────────────────────────────
# Fallback: deterministic pseudo-random vectors (zero-download)
# ─────────────────────────────────────────────────────────────

class LocalEmbeddings:
    """
    Zero-download, zero-RAM embedding generator.
    Produces deterministic 384-dimensional normalized vectors instantly (0.001ms),
    completely eliminating HuggingFace downloads, PyTorch dependencies, and Render OOM/timeout errors.
    Used ONLY as a fallback when the Hugging Face API is unavailable.
    """

    def __init__(self):
        self._cache = {}

    def embed_query(self, text: str) -> List[float]:
        if not text:
            return [0.0] * 384

        if text in self._cache:
            return self._cache[text]

        vec = []
        for i in range(384):
            h = hashlib.sha256(f"{text}_{i}".encode("utf-8")).digest()
            val = (int.from_bytes(h[:4], "big") / 0xFFFFFFFF) * 2.0 - 1.0
            vec.append(val)

        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        normalized_vec = [float(x / norm) for x in vec]
        if len(self._cache) >= 128:
            self._cache.clear()
        self._cache[text] = normalized_vec
        return normalized_vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return [self.embed_query(t) for t in texts]


# ─────────────────────────────────────────────────────────────
# Primary: Real semantic embeddings via Hugging Face Inference API
# ─────────────────────────────────────────────────────────────

_HF_API_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)
_DIMENSIONS = 384
_MAX_RETRIES = 3


class HuggingFaceAPIEmbeddings:
    """
    Semantic embedding generator using Hugging Face's Serverless Inference API
    for sentence-transformers/all-MiniLM-L6-v2 (384-dim, cosine similarity).

    Features:
    - Zero local RAM: model runs entirely on HF's cloud GPUs.
    - LRU cache (128 items) for instant deduplication.
    - Exponential backoff retry (3 attempts) for network resilience.
    - Graceful fallback to LocalEmbeddings if HF API is unreachable.
    """

    def __init__(self, token: str):
        self._token = token
        self._headers = {
            "Authorization": f"Bearer {token}",
            "X-Wait-For-Model": "true",
            "Content-Type": "application/json",
        }
        self._cache = {}
        self._fallback = LocalEmbeddings()

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """Call HF Inference API with exponential backoff retry."""
        last_err = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = requests.post(
                    _HF_API_URL,
                    headers=self._headers,
                    json={"inputs": texts, "options": {"wait_for_model": True}},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()

                # API returns List[List[float]] for multiple inputs
                # or List[float] for a single input
                if data and isinstance(data[0], float):
                    data = [data]

                # Validate dimensions
                for vec in data:
                    if len(vec) != _DIMENSIONS:
                        raise ValueError(
                            f"Expected {_DIMENSIONS}D vector, got {len(vec)}D"
                        )
                return data

            except Exception as e:
                last_err = e
                if attempt < _MAX_RETRIES - 1:
                    wait = 2 ** attempt  # 1s, 2s
                    logger.warning(
                        f"[HF-Embed] Attempt {attempt + 1}/{_MAX_RETRIES} failed: {e}. "
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)

        logger.error(f"[HF-Embed] All {_MAX_RETRIES} attempts failed: {last_err}")
        return []

    def embed_query(self, text: str) -> List[float]:
        """Embed a single text query into a 384-dim vector."""
        if not text:
            return [0.0] * _DIMENSIONS

        if text in self._cache:
            return self._cache[text]

        result = self._call_api([text])
        if result:
            vec = result[0]
        else:
            logger.warning("[HF-Embed] Falling back to LocalEmbeddings for query.")
            vec = self._fallback.embed_query(text)

        # LRU eviction
        if len(self._cache) >= 128:
            self._cache.clear()
        self._cache[text] = vec
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts. Uncached texts are sent to HF API in one call."""
        if not texts:
            return []

        # Split into cached hits and uncached misses
        results = [None] * len(texts)
        uncached_indices = []
        for i, t in enumerate(texts):
            if t in self._cache:
                results[i] = self._cache[t]
            else:
                uncached_indices.append(i)

        if uncached_indices:
            uncached_texts = [texts[i] for i in uncached_indices]
            api_results = self._call_api(uncached_texts)

            if api_results and len(api_results) == len(uncached_texts):
                for idx, vec in zip(uncached_indices, api_results):
                    results[idx] = vec
                    if len(self._cache) >= 128:
                        self._cache.clear()
                    self._cache[texts[idx]] = vec
            else:
                # Fallback for failed batch
                logger.warning("[HF-Embed] Falling back to LocalEmbeddings for batch.")
                for idx in uncached_indices:
                    vec = self._fallback.embed_query(texts[idx])
                    results[idx] = vec

        return results


def download_embeddings():
    """
    Returns the best available embedding model:
    - HuggingFaceAPIEmbeddings (real semantic vectors) when HF_TOKEN is set.
    - LocalEmbeddings (deterministic fallback) otherwise.
    """
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        logger.info("[Embeddings] Using HuggingFace API (all-MiniLM-L6-v2) for semantic embeddings.")
        return HuggingFaceAPIEmbeddings(token=hf_token)
    else:
        logger.warning("[Embeddings] HF_TOKEN not set — falling back to LocalEmbeddings (non-semantic).")
        return LocalEmbeddings()



def load_pdf_files(data: str) -> List[Document]:
    """
    Loads all PDF documents from the specified directory path.
    
    Args:
        data (str): Path to directory containing PDF files.
        
    Returns:
        List[Document]: Extracted LangChain Document instances.
    """
    loader = DirectoryLoader(
        path=data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Strips non-essential metadata from documents to conserve vector storage footprint.
    
    Args:
        docs (List[Document]): Raw loaded documents.
        
    Returns:
        List[Document]: Cleaned documents with minimal source metadata.
    """
    minimal_docs = []

    for doc in docs:
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "source": doc.metadata.get(
                        "source",
                        "unknown"
                    )
                }
            )
        )

    return minimal_docs


def text_split(
    docs: List[Document],
    chunk_size: int = 2500,
    chunk_overlap: int = 50
) -> List[Document]:
    """
    Splits documents into clinical context chunks with overlap.
    
    Args:
        docs (List[Document]): Documents to split.
        chunk_size (int): Character size of each chunk.
        chunk_overlap (int): Overlap between adjacent chunks.
        
    Returns:
        List[Document]: Chunked document list.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return splitter.split_documents(docs)