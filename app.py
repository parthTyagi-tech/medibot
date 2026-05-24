from flask import (
    Flask,
    render_template,
    jsonify,
    request
)

from research.src.helper import (
    download_embeddings
)

from langchain_pinecone import (
    PineconeVectorStore
)

from langchain_openai import (
    ChatOpenAI
)

from langchain.chains import (
    create_retrieval_chain
)

from langchain.chains.combine_documents import (
    create_stuff_documents_chain
)

from langchain_core.prompts import (
    ChatPromptTemplate
)

from dotenv import (
    load_dotenv
)

from research.src.prompt import *

import os

app = Flask(__name__)
 
load_dotenv()
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")


os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY

embedding = download_embeddings()
index_name="medical-chatbot"
docsearch=PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
    
)


# connecting to LLM 
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
from langchain_openai import ChatOpenAI

chatModel = ChatOpenAI(
    model="gpt-4o",
    
)

question_answer_chain=create_stuff_documents_chain(chatModel,prompt)
rag_chain=create_retrieval_chain(retriever,question_answer_chain)

@app.route(
    "/get",
    methods=["GET", "POST"]
)
def chat():

    msg = request.form["msg"]

    print(
        "User:",
        msg
    )

    response = (
        rag_chain.invoke(
            {
                "input": msg
            }
        )
    )

    answer = (
        response["answer"]
    )

    print(
        "Response:",
        answer
    )

    return str(
        answer
    )


@app.route("/")
def index():
    return render_template('chat.html')

if __name__== "__main__":
  app.run(host="0.0.0.0",debug=True)