from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import requests
import time


class LocalEmbeddings:

    def __init__(self):
        # We define a primary and secondary fallback endpoint for DNS resilience
        self.api_urls = [
            "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
            "https://api.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
            "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        ]
        self.headers = {}
        token = os.getenv("HF_TOKEN")
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    def _query(self, texts: List[str]) -> List[List[float]]:
        last_err = None
        for api_url in self.api_urls:
            try:
                response = requests.post(
                    api_url,
                    headers=self.headers,
                    json={"inputs": texts, "options": {"wait_for_model": True}},
                    timeout=12
                )
                res_json = response.json()
                
                # If the model is currently loading, retry once after waiting
                if isinstance(res_json, dict) and "estimated_time" in res_json:
                    wait_time = min(float(res_json["estimated_time"]), 10.0)
                    time.sleep(wait_time)
                    response = requests.post(
                        api_url,
                        headers=self.headers,
                        json={"inputs": texts, "options": {"wait_for_model": True}},
                        timeout=12
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
                            return pooled
                        else:
                            return res_json
                if isinstance(res_json, dict) and "error" in res_json:
                    raise ValueError(res_json["error"])
                raise ValueError(f"Unexpected response format from Hugging Face: {res_json}")
            except Exception as e:
                last_err = e
                print(f"HF query failed for {api_url}: {e}")
                continue
                
        # If all fail, let's output a fallback log and return dummy vectors to prevent crashes
        print("HF Embedding Error (all endpoints failed):", last_err)
        return [[0.0] * 384 for _ in texts]

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