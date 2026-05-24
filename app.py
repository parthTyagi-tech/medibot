from flask import Flask, render_template, request

from research.src.helper import download_embeddings
from research.src.prompt import prompt

from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv
import os


# ==========================
# Flask App
# ==========================
app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# ==========================
# Embeddings + Pinecone
# ==========================
embedding = download_embeddings()

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# ==========================
# LLM
# ==========================
chatModel = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7
)


# ==========================
# RAG Chain
# ==========================
question_answer_chain = create_stuff_documents_chain(
    chatModel,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)


# ==========================
# Detect Casual Chat
# ==========================
def is_medical_query(text):

    text = text.lower()

    medical_keywords = [
        "pain",
        "fever",
        "disease",
        "medicine",
        "symptom",
        "infection",
        "treatment",
        "doctor",
        "health",
        "virus",
        "blood",
        "headache",
        "hepatitis",
        "diabetes",
        "cancer",
        "covid",
        "body",
        "stomach",
        "medical"
    ]

    return any(
        word in text
        for word in medical_keywords
    )


# ==========================
# Chat Endpoint
# ==========================
@app.route(
    "/get",
    methods=["GET", "POST"]
)
def chat():

    msg = request.form["msg"]

    print("User:", msg)

    try:

        if is_medical_query(msg):

            response = rag_chain.invoke({
                "input": msg
            })

            answer = response["answer"]

        else:

            response = chatModel.invoke(msg)

            answer = response.content

        print("Bot:", answer)

        return answer

    except Exception as e:

        print(e)

        return "Something went wrong."


# ==========================
# Homepage
# ==========================
@app.route("/")
def index():
    return render_template("chat.html")


# ==========================
# Run
# ==========================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )