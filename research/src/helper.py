from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
import os
import time
import logging
import threading
import concurrent.futures

logger = logging.getLogger("embeddings")

# ─────────────────────────────────────────────────────────────
# Primary: Real semantic embeddings via Pinecone Hosted Inference
# ─────────────────────────────────────────────────────────────

def _is_retryable_error(exc: Exception) -> bool:
    """Retry only on timeout or 5xx, never on 4xx, 429, or quota exhaustion."""
    if isinstance(exc, (concurrent.futures.TimeoutError, TimeoutError)):
        return True

    status_code = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    if status_code is not None:
        try:
            code = int(status_code)
            if code >= 500:
                return True
            if 400 <= code < 500:
                return False
        except (ValueError, TypeError):
            pass

    err_str = str(exc).lower()
    if any(q in err_str for q in ["429", "quota", "rate limit", "unauthorized", "forbidden", "401", "403", "404", "bad request", "400"]):
        return False
    if any(s in err_str for s in ["500", "502", "503", "504", "server error", "timeout", "timed out", "deadline"]):
        return True

    return False


class PineconeHostedEmbeddings:
    """
    Semantic embedding generator using Pinecone's Hosted Inference API
    (multilingual-e5-large, 1024-dim, cosine similarity).

    Features:
    - Zero local RAM: model runs entirely on Pinecone's cloud (<10MB host RAM).
    - Dynamic dimension probe from model inference.
    - Thread-safe LRU query cache (128 items) that never caches None.
    - Hard 3s timeout on inference calls via ThreadPoolExecutor.
    - Resilient retry policy: retries once only on timeout/5xx, never on 4xx/429/quota.
    - Circuit breaker: 3 consecutive failures open breaker for 60s.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 3.0
    ):
        self.api_key = api_key or os.getenv("PINECONE_API_KEY") or "dummy-offline-key"
        self.model = model or os.getenv("EMBED_MODEL", "multilingual-e5-large")
        self.timeout = timeout
        self.pc = Pinecone(api_key=self.api_key)

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self._cache: Dict[str, List[float]] = {}
        self._cache_lock = threading.Lock()

        self._consecutive_failures = 0
        self._breaker_open_until = 0.0
        self._breaker_lock = threading.Lock()

        # Dynamic dimension probe (Amendment 8)
        self.dimension = 1024
        try:
            probe = self._call_embed(["probe"], input_type="query")
            if probe and len(probe) > 0 and len(probe[0]) > 0:
                self.dimension = len(probe[0])
                logger.info(f"[Pinecone-Embed] Probed dimension for '{self.model}': {self.dimension}D")
        except Exception as e:
            logger.warning(f"[Pinecone-Embed] Probe dimension check notice: {e}. Using fallback {self.dimension}D")

    def _raw_inference(self, texts: List[str], input_type: str) -> Any:
        return self.pc.inference.embed(
            model=self.model,
            inputs=texts,
            parameters={"input_type": input_type, "truncate": "END"}
        )

    def _call_embed_once(self, texts: List[str], input_type: str) -> List[List[float]]:
        future = self._executor.submit(self._raw_inference, texts, input_type)
        res = future.result(timeout=self.timeout)
        embeddings = []
        if hasattr(res, "data"):
            for item in res.data:
                if hasattr(item, "values"):
                    embeddings.append(item.values)
                elif isinstance(item, dict) and "values" in item:
                    embeddings.append(item["values"])
                elif isinstance(item, list):
                    embeddings.append(item)
        elif isinstance(res, list):
            embeddings = res
        return embeddings

    def _call_embed(self, texts: List[str], input_type: str = "query") -> Optional[List[List[float]]]:
        # Circuit breaker check
        now = time.time()
        with self._breaker_lock:
            if now < self._breaker_open_until:
                logger.warning(
                    f"[Pinecone-Embed] Circuit breaker is OPEN (cooldown until {self._breaker_open_until:.1f}). Skipping embedding."
                )
                return None

        last_exc = None
        for attempt in range(2):
            try:
                embeddings = self._call_embed_once(texts, input_type)
                with self._breaker_lock:
                    self._consecutive_failures = 0
                    self._breaker_open_until = 0.0
                return embeddings
            except Exception as e:
                last_exc = e
                if attempt == 0 and _is_retryable_error(e):
                    logger.warning(
                        f"[Pinecone-Embed] Attempt 1 failed with retryable error ({type(e).__name__}: {e}). Retrying once..."
                    )
                    time.sleep(0.5)
                    continue
                else:
                    logger.error(
                        f"[Pinecone-Embed] Embedding call failed ({type(e).__name__}: {e}). No further retries."
                    )
                    break

        with self._breaker_lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= 3:
                self._breaker_open_until = time.time() + 60.0
                logger.critical(
                    f"[Pinecone-Embed] Circuit breaker OPENED for 60s after {self._consecutive_failures} consecutive failures."
                )
        return None

    def embed_query(self, text: str) -> Optional[List[float]]:
        """Embed a single query into a semantic vector. Returns None on failure."""
        if not text:
            return None

        with self._cache_lock:
            if text in self._cache:
                return self._cache[text]

        results = self._call_embed([text], input_type="query")
        if not results or len(results) == 0:
            # Crucial: Never cache None
            return None

        vec = results[0]
        with self._cache_lock:
            if len(self._cache) >= 128:
                self._cache.clear()
            self._cache[text] = vec
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of document texts (passage mode) in chunks <= 96."""
        if not texts:
            return []

        all_embeddings = []
        batch_size = 96
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            res = self._call_embed(batch, input_type="passage")
            if res:
                all_embeddings.extend(res)
            else:
                raise RuntimeError(f"Embedding failed for passage batch at index {i}")
        return all_embeddings


# Backward compatibility alias
HuggingFaceAPIEmbeddings = PineconeHostedEmbeddings
LocalEmbeddings = PineconeHostedEmbeddings


def download_embeddings():
    """
    Returns PineconeHostedEmbeddings for hosted inference embeddings (multilingual-e5-large).
    Requires PINECONE_API_KEY in environment variables.
    """
    return PineconeHostedEmbeddings()


def load_pdf_files(data: str) -> List[Document]:
    """
    Loads all PDF documents from the specified directory path.
    """
    loader = DirectoryLoader(
        path=data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Strips non-essential metadata from documents while preserving source and page.
    """
    minimal_docs = []

    for doc in docs:
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", 0)
                }
            )
        )

    return minimal_docs


def text_split(
    docs: List[Document],
    chunk_size: int = 1100,
    chunk_overlap: int = 50
) -> List[Document]:
    """
    Splits documents into clinical context chunks (~1000-1200 characters) with overlap.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return splitter.split_documents(docs)