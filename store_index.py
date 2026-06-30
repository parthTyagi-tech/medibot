from research.src.helper import (
    load_pdf_files,
    filter_to_minimal_docs,
    text_split,
    download_embeddings
)

from dotenv import load_dotenv
import os


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Load PDFs
extracted_data = load_pdf_files("data")

# Filter
filter_doc = filter_to_minimal_docs(extracted_data)

# Split
text_chunk = text_split(filter_doc)

print(f"Number of chunks: {len(text_chunk)}")

# Embeddings
embedding = download_embeddings()

# Import native Pinecone client
from pinecone import Pinecone

print("Upserting chunks to Pinecone using native client...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("medical-chatbot")

vectors = []
for i, chunk in enumerate(text_chunk):
    text = chunk.page_content
    # Generate embedding query vector
    vector = embedding.embed_query(text)
    
    metadata = chunk.metadata.copy()
    metadata["text"] = text  # Critical: Store text in metadata so CustomPineconeRetriever can read it
    
    vectors.append({
        "id": f"chunk-{i}",
        "values": vector,
        "metadata": metadata
    })
    
    # Batch upsert in sizes of 100
    if len(vectors) >= 100:
        index.upsert(vectors=vectors)
        vectors = []

if vectors:
    index.upsert(vectors=vectors)

print("Index successfully stored in Pinecone.")