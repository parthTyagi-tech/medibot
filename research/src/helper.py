from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import requests
import time


class LocalEmbeddings:

    def __init__(self):
        # Current valid Hugging Face Router & Inference API endpoints
        self.api_urls = [
            "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2",
            "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
            "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
        ]
        self.headers = {}
        token = os.getenv("HF_TOKEN")
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        self._cache = {}

    def _query(self, texts: List[str]) -> List[List[float]]:
        # Check cache for single text queries
        if len(texts) == 1 and texts[0] in self._cache:
            return [self._cache[texts[0]]]

        last_err = None
        for api_url in self.api_urls:
            try:
                response = requests.post(
                    api_url,
                    headers=self.headers,
                    json={"inputs": texts, "options": {"wait_for_model": True}},
                    timeout=3
                )
                res_json = response.json()
                
                # If the model is currently loading, retry once after a short wait
                if isinstance(res_json, dict) and "estimated_time" in res_json:
                    wait_time = min(float(res_json["estimated_time"]), 3.0)
                    time.sleep(wait_time)
                    response = requests.post(
                        api_url,
                        headers=self.headers,
                        json={"inputs": texts, "options": {"wait_for_model": True}},
                        timeout=3
                    )
                    res_json = response.json()
                
                if isinstance(res_json, list):
                    # Handle sequence token embedding outputs [batch, seq, dim] -> mean pooling
                    if len(res_json) > 0 and isinstance(res_json[0], list):
                        if isinstance(res_json[0][0], list):
                            pooled = []
                            for doc_emb in res_json:
                                seq_len = len(doc_emb)
                                dim = len(doc_emb[0])
                                mean_vector = [sum(doc_emb[t][d] for t in range(seq_len)) / seq_len for d in range(dim)]
                                pooled.append(mean_vector)
                            res_vecs = pooled
                        else:
                            res_vecs = res_json

                        if len(texts) == 1 and len(res_vecs) > 0:
                            self._cache[texts[0]] = res_vecs[0]
                        return res_vecs
                if isinstance(res_json, dict) and "error" in res_json:
                    raise ValueError(res_json["error"])
                raise ValueError(f"Unexpected response format from Hugging Face: {res_json}")
            except Exception as e:
                last_err = e
                print(f"HF query failed for {api_url}: {e}")
                continue
                
        # If all fail, return fallback zero vectors fast to prevent blocking backend
        print("HF Embedding Error (all endpoints failed):", last_err)
        dummy_vecs = [[0.0] * 384 for _ in texts]
        if len(texts) == 1:
            self._cache[texts[0]] = dummy_vecs[0]
        return dummy_vecs

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            embeddings.extend(self._query(batch))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._query([text])[0]


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