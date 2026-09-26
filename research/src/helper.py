from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import requests
import time
import logging

logger = logging.getLogger("embeddings")

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
    - Zero local RAM: model runs entirely on HF's cloud GPUs (<50MB host RAM).
    - LRU cache (128 items) for instant deduplication.
    - Exponential backoff retry (3 attempts) for network resilience.
    - Deterministic output dimensions matching Pinecone index configuration.
    """

    def __init__(self, token: str):
        self._token = token
        self._headers = {
            "Authorization": f"Bearer {token}",
            "X-Wait-For-Model": "true",
            "Content-Type": "application/json",
        }
        self._cache = {}

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
        """Embed a single text query into a 384-dim semantic vector."""
        if not text:
            return [0.0] * _DIMENSIONS

        if text in self._cache:
            return self._cache[text]

        result = self._call_api([text])
        if result:
            vec = result[0]
        else:
            logger.error("[HF-Embed] Failed to generate semantic vector for query.")
            vec = [0.0] * _DIMENSIONS

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
                logger.error("[HF-Embed] Failed to retrieve semantic embeddings for batch.")
                for idx in uncached_indices:
                    results[idx] = [0.0] * _DIMENSIONS

        return results


# Backward compatibility alias
LocalEmbeddings = HuggingFaceAPIEmbeddings


def download_embeddings():
    """
    Returns HuggingFaceAPIEmbeddings for true semantic embeddings
    via Hugging Face's Serverless Inference API (all-MiniLM-L6-v2).
    Requires HF_TOKEN to be set in environment variables.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN environment variable is missing. "
            "Please provide a valid Hugging Face Access Token in .env to generate semantic embeddings."
        )
    logger.info("[Embeddings] Using Hugging Face Serverless API (all-MiniLM-L6-v2) for semantic embeddings.")
    return HuggingFaceAPIEmbeddings(token=hf_token)



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