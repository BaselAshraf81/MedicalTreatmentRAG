# model.py

import os
import re
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA, create_retrieval_chain
from langchain.memory import ConversationBufferMemory
import torch
import gc

# Save the Hugging Face token (ensure this is handled securely in production)
from huggingface_hub.hf_api import HfFolder
HfFolder.save_token('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')

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def preprocess_text(text):
    # Replace line breaks followed by bullets with spaces
    processed_text = re.sub(r'\n•\n', ' • ', text)
    # Remove leading and trailing spaces
    processed_text = processed_text.strip()
    # Replace multiple spaces with a single space
    processed_text = re.sub(r'\s+', ' ', processed_text)
    return processed_text

def split_sections(text):
    # Split the text into sections based on section headers
    sections = re.split(r'\n([A-Za-z &]+)\n', text)
    # Remove empty strings from the list
    sections = [section.strip() for section in sections if section.strip()]

    formatted_text = {}
    # Iterate over sections
    for i in range(0, len(sections), 2):
        section_name = sections[i]
        section_content = sections[i+1] if i+1 < len(sections) else ""
        formatted_text[section_name] = section_content.split(' • ')

    return formatted_text

def format_text(formatted_text):
    # Create the final formatted output
    output = ""
    for section, items in formatted_text.items():
        output += f"### {section}\n\n"
        for item in items:
            if item.strip():
                output += f"- {item.strip()}\n"
        output += "\n"
    return output

def process_text(text):
    processed_text = preprocess_text(text)
    formatted_text = split_sections(processed_text)
    output = format_text(formatted_text)
    return output

def initialize_chain(diagnostic_path, doctors_recom):
    # Extract and process texts
    pdf_diagnostic = extract_text_from_pdf(diagnostic_path)
    pdf_doctors = extract_text_from_pdf(doctors_recom)

    diagnostic_text = process_text(pdf_diagnostic)
    doctors_text = process_text(pdf_doctors)

    # Split texts into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    chunks = text_splitter.split_text(diagnostic_text)
    chunks2 = text_splitter.split_text(doctors_text)

    documents = [Document(page_content=chunk) for chunk in chunks]
    documents2 = [Document(page_content=chunk) for chunk in chunks2]

    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create vector store and add documents
    vector_store = Chroma.from_documents(documents, embeddings)
    vector_store.add_documents(documents2)
    retriever = vector_store.as_retriever()

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.4,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    # Define system prompt
    system_prompt = (
        """
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

        **Recommended Doctors**:

        **Chest Doctors:**
        - Amr Maher El-Tounsy: Sh. El-Haram, Giza Square, Nasr Building
        - Mohamed Fawzy Badreldin Abbas: Marad Tower, El-Giza

        **Bone Doctors:**
        - Adel Adawy: Tahrir Square, Dokki Administrative Tower
        - Abdel Rahman El-Beshbeshy: 34 Sh. Port Said, in front of Ahmed Maher Hospital
        - Atef Mohamed Morsi: Prima Vista Tower, Magda Square, 6th of October
        - Khaled El-Sherebiny: 26th of July St., Sphinx, Mohandessin
        - Fadi Michel Fahmy: Sinan St., Gesr El-Suez, El-Zeitoun

        **Cancer Doctors:**
        - Ahmed Bakir: Sh. Al-Manial
        - Mohamed Ahmed Abdel Hamid: 51 Sh. Al-Tahrir, Dokki
        - Mohamed Attia Al-Kurdi: Sh. Al-Lasalky, Maadi, near Moamen
        - Ali Zidan Tahami: Sh. Al-Maadi, near Al-Shabrawi, Maadi
        - Ramz Abd El-Masih: Sh. Osman Ibn Affan, Ismailia Square, Cairo
        - Amr Kamel: Sh. Sudan, Mohandessin

        **Brain Doctors:**
        - Mostafa Kamel El-Fouly: Sh. Mostafa El-Nahhas, Doctors Tower, Nasr City

        **Heart Doctors:**
        - Ibrahim Ahmed Mostafa: Protomedical Unit 5, First Settlement
        - Ahmed Hussein Mahmoud: Sh. Misr Helwan Agricultura, Dar El-Salam
        - Hossam Abdel-Aleem Shaheen: Sh. Qawmeya Arabia, Imbaba
        - Khaled El-Tahamy Moharread: Tower, Faisal Road, in front of College of Physical Education
        - Nabil Sayed Hamida Ammar: Sh. El-Nadi El-Riyadi, Doctors Tower, Faisal
        - Samir Saber Mohamed Ash: Hassan El-Anwar, King Salah Underpass, Misr El-Qadima

        Be sure to:
        - Use simple, patient-friendly language that is easy to understand.
        - Locate the disease in the body, helping the patient understand how it affects them.
        - Provide doctors that specialize in the condition and their location for easy access.
        - You don't have to include all the doctors in the list,
        - Tailor your recommendations and advice to the patient's specific condition.
        - Don't write "see PDF for contact information."
        _ Be concise

        You have access to the following context to help you respond: {context}
        """
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Initialize the retrieval chain
    retrieval_chain = create_retrieval_chain(llm, retriever, prompt=prompt)

    return retrieval_chain

# Initialize the chain (this can be called once when the app starts)
def get_chain():
    diagnostic_path = 'Dataset.pdf'          # Ensure these PDFs are in the project directory
    doctors_recom = 'final_organized_doctors_list.pdf'
    chain = initialize_chain(diagnostic_path, doctors_recom)
    return chain
