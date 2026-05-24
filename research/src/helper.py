from typing import List

from langchain.schema import Document

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)


# Import embeddings lazily
def download_embeddings():

    from langchain_huggingface import (
        HuggingFaceEmbeddings
    )

    return HuggingFaceEmbeddings(
        model_name=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        )
    )


# Extract PDFs
def load_pdf_files(data):

    loader = DirectoryLoader(
        path=data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    return documents


# Reduce metadata
def filter_to_minimal_docs(
    docs: List[Document]
):

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


# Split documents
def text_split(
    docs,
    chunk_size=2500,
    chunk_overlap=50
):

    splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    )

    return splitter.split_documents(
        docs
    )


# Debug only
if __name__ == "__main__":

    print(
        "helper.py loaded successfully"
    )