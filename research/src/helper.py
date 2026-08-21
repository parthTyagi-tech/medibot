from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import requests
import time


class LocalEmbeddings:
    """
    High-performance, reliable embedding generator using sentence-transformers/all-MiniLM-L6-v2.
    First tries langchain_huggingface HuggingFaceEmbeddings / SentenceTransformer (local, 0ms network latency),
    with fallback to HuggingFace Router API if needed.
    """

    def __init__(self):
        self._hf_embedder = None
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            self._hf_embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            print("[Embeddings] Local HuggingFaceEmbeddings initialization notice:", e)

        self.api_urls = [
            "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2",
        ]
        self.headers = {}
        token = os.getenv("HF_TOKEN")
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        self._cache = {}

    def embed_query(self, text: str) -> List[float]:
        if not text:
            return [0.0] * 384
        if self._hf_embedder:
            try:
                return self._hf_embedder.embed_query(text)
            except Exception as e:
                print("[Embeddings] Local embed_query fallback:", e)

        if text in self._cache:
            return self._cache[text]

        for api_url in self.api_urls:
            try:
                response = requests.post(
                    api_url,
                    headers=self.headers,
                    json={"inputs": [text], "options": {"wait_for_model": True}},
                    timeout=5
                )
                res_json = response.json()
                if isinstance(res_json, list) and len(res_json) > 0:
                    vec = res_json[0]
                    if isinstance(vec, list) and len(vec) == 384:
                        self._cache[text] = vec
                        return vec
            except Exception:
                pass

        return [0.0] * 384

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self._hf_embedder:
            try:
                return self._hf_embedder.embed_documents(texts)
            except Exception as e:
                print("[Embeddings] Local embed_documents fallback:", e)
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