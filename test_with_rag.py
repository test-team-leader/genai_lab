from langchain_community.chat_models import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain_commutity.embeddings import OllamaEmbeddings

import streamlit as st

llm = ChatOllama(model="gpt-oss:20b")  # 로컬에 설치된 모델 사용

def load_and_split_pdf(file_path)
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(_docs)
    vector_store = Chroma.from_documents(split_docs, OllamaEmbeddings(model="mxbai-embed-large:335m"))
    return vector_store

def initialize_components(selected_model):
    file_path = r"E:\genai-lab\streamlit\test\example.pdf"
    pages = load_and_split_pdf(file_path)
    vector_store = create_vector_store(pages)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    contextualize_q_system_prompt = ("""Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is.""")
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
