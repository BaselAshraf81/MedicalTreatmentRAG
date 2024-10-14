# app.py

import os

# Set the environment variable before importing any other libraries
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
from model import get_chain

# Initialize the QA chain with caching to improve performance
@st.cache_resource
def load_chain():
    chain = get_chain()
    return chain

def main():
    st.title("Medical Consultation AI")
    st.write("Ask your medical questions and receive expert advice.")

    # File uploader for diagnostic and doctors list PDFs
    diagnostic_file = st.file_uploader("Upload your diagnostic PDF", type=["pdf"])
    doctors_file = st.file_uploader("Upload the doctors list PDF", type=["pdf"])

    if diagnostic_file and doctors_file:
        # Save uploaded files to the current directory
        with open("Dataset.pdf", "wb") as f:
            f.write(diagnostic_file.getbuffer())
        with open("final_organized_doctors_list.pdf", "wb") as f:
            f.write(doctors_file.getbuffer())

        # Load the chain
        chain = load_chain()

        # User input for symptoms
        user_input = st.text_area("Describe your symptoms or concerns:")
        if st.button("Get Diagnosis"):
            if user_input.strip() == "":
                st.error("Please enter your symptoms or concerns.")
            else:
                with st.spinner("Processing..."):
                    try:
                        answer = chain.invoke({"input": user_input})
                        st.markdown(answer["answer"])
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

# Run the main function
if __name__ == "__main__":
    main()

# Hide the footer (optional)
st.markdown("""
    <style>
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
