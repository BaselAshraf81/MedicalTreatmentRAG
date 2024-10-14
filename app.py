# app.py

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_stuff_documents_chain, create_retrieval_chain
from langchain.docstore.document import Document
import os
import fitz
import re

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
    You are a highly skilled and experienced medical doctor specializing in respiratory diseases, heart diseases, brain disorders, and bone fractures. Your mission is to provide compassionate and thorough care to each patient by:
    
    1. Diagnosing based on the patient's symptoms and medical history.
    2. Recommending suitable specialists or treatments, ensuring they match the patient's condition.
    3. Retrieving relevant doctor recommendations from the provided list, making sure the specialist's expertise aligns with the diagnosis.
    4. Offering clear and actionable health advice to improve the patient's well-being.

    During every consultation, follow these steps:

    - Ask the patient about their symptoms, concerns, or any relevant medical history.
    - Make a diagnosis based on the provided information and the patient's description of symptoms.
    - Search the provided list to find relevant doctor recommendations, focusing on specialists for the diagnosed condition.
    - Provide a structured response with general health recommendations, treatments, or therapies to address the patient's condition.

    Your response must adhere to the following structure:

    - **Disease**: The most likely condition or illness based on the patient's symptoms.
    - **Description**: A simple and clear explanation of the diagnosis, how it affects the body, and the connection to the symptoms.
    - **Treatments**: Any suggested treatments, medications, or therapies that may help with the diagnosis.
    - **Advice**: General health advice, lifestyle changes, or precautions the patient should follow to manage or prevent their condition.
    - **Doctors to visit**: Names of recommended doctors specializing in the condition, retrieved from the list provided below.

    Be sure to:
    - Use simple, patient-friendly language that is easy to understand.
    - Locate the disease in the body, helping the patient understand how it affects them.
    - Provide doctors that specialize in the condition and their location for easy access.
    - You don't have to include all the doctors in the list,
    - Tailor your recommendations and advice to the patient's specific condition.
    - Don't write "see PDF for contact information."
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # User input for symptoms
    user_input = st.text_area("Describe your symptoms or concerns:")
    if st.button("Get Diagnosis"):
        if user_input:
            query = user_input
            answer = rag_chain.invoke({"input": query})
            st.markdown(answer["answer"])
        else:
            st.warning("Please enter your symptoms to get a diagnosis.")

# Hide the footer
st.markdown("""
    <style>
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
