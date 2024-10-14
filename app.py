# app.py

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_qa_chain  # Updated import
from langchain.docstore.document import Document
import os
import fitz

# Import your previously defined functions
from model import extract_text_from_pdf, preprocess_text, split_sections, format_text, process_text, rag_chain

# Set up Streamlit app
st.title("Medical Consultation Chatbot")

# File uploader for diagnostic and doctors list PDFs
diagnostic_file = st.file_uploader("Upload your diagnostic PDF", type=["pdf"])
doctors_file = st.file_uploader("Upload the doctors list PDF", type=["pdf"])

if diagnostic_file and doctors_file:
    # Process the uploaded PDFs
    diagnostic_text = process_text(extract_text_from_pdf(diagnostic_file))
    doctors_text = process_text(extract_text_from_pdf(doctors_file))

    # Create document chunks and vector store
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    chunks = text_splitter.split_text(diagnostic_text)
    chunkkk = text_splitter.split_text(doctors_text)

    documents = [Document(page_content=chunk) for chunk in chunks]
    documents2 = [Document(page_content=chunk) for chunk in chunkkk]

    # Initialize embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_documents(documents, embeddings)
    vector_store.add_documents(documents2)
    retriever = vector_store.as_retriever()

    # Initialize the LLM for QA
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.4,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    # Define the system prompt
    system_prompt = """
    You are a highly skilled and experienced medical doctor specializing in respiratory diseases, heart diseases, brain disorders, and bone fractures...
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Initialize the retrieval QA chain
    qa_chain = create_retrieval_qa_chain(llm, retriever)

    # User input for symptoms
    user_input = st.text_area("Describe your symptoms or concerns:")
    if st.button("Get Diagnosis"):
        if user_input:
            query = user_input
            answer = qa_chain.invoke({"input": query})  # Use qa_chain instead of rag_chain
            st.markdown(answer["answer"])
        else:
            st.warning("Please enter your symptoms to get a diagnosis.")

# Hide the footer
st.markdown("""
    <style>
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
