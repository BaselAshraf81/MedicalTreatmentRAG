# app.py

import os

# Set the environment variable first
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def main():
    st.title("ChromaDB Test")
    st.write("Testing ChromaDB initialization.")
    
    # Dummy documents
    documents = [Document(page_content="This is a test document.")]
    
    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    try:
        # Initialize Chroma
        vector_store = Chroma.from_documents(documents, embeddings)
        st.success("ChromaDB initialized successfully.")
    except Exception as e:
        st.error(f"Error initializing ChromaDB: {e}")

if __name__ == "__main__":
    main()
