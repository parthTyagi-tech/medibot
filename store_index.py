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

# Import Pinecone Vector Store
from langchain_pinecone import PineconeVectorStore

print("Upserting chunks to Pinecone...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunk,
    embedding=embedding,
    index_name="medical-chatbot"
)
print("Index successfully stored in Pinecone.")