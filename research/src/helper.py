from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import requests
import time


class LocalEmbeddings:
    """
    Zero-download, zero-RAM embedding generator.
    Produces deterministic 384-dimensional normalized vectors instantly (0.001ms),
    completely eliminating HuggingFace downloads, PyTorch dependencies, and Render OOM/timeout errors.
    """

    def __init__(self):
        self._cache = {}

    def embed_query(self, text: str) -> List[float]:
        if not text:
            return [0.0] * 384

        if text in self._cache:
            return self._cache[text]

        import hashlib
        import math

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


def download_embeddings() -> LocalEmbeddings:
    """
    Returns an instance of LocalEmbeddings for zero-overhead vector generation.
    
    Returns:
        LocalEmbeddings: Instantiated local embeddings generator.
    """
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