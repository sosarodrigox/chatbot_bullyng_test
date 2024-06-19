import streamlit as st

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# Loas openai api key
from dotenv import load_dotenv

# load environment variables:
load_dotenv()

st.header("Mapa Nacional Bullying Chat V1.0 Test ")
pdf_obj = st.file_uploader("Cargar PDF", type=["pdf"])


def create_embeddings(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )

    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base


if pdf_obj:
    knowledge_base = create_embeddings(pdf_obj)
    user_question = st.text_input("Realiza tu pregunta al PDF acá:")

    if user_question:
        docs = knowledge_base.similarity_search(user_question, 3)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo")
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=user_question)

        st.write(response)
