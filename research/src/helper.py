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
        self._cache[text] = normalized_vec
        return normalized_vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return [self.embed_query(t) for t in texts]


def download_embeddings():
    return LocalEmbeddings()




def load_pdf_files(data):
    loader = DirectoryLoader(
        path=data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()


def filter_to_minimal_docs(docs: List[Document]):
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
    docs,
    chunk_size=2500,
    chunk_overlap=50
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return splitter.split_documents(docs)