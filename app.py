import streamlit as st
from model import get_chain
import subprocess
import os

# Function to run bash commands to export environment variables
def export_environment_variables():
    try:
        # Run export commands as bash commands
        subprocess.run("export GOOGLE_API_KEY='AIzaSyBadUb2oZd7KjS8eY6XH8-AbMhO48nEs0g'", shell=True, executable="/bin/bash")
        subprocess.run("export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'", shell=True, executable="/bin/bash")
        st.success("Environment variables set successfully in the bash shell.")
    except Exception as e:
        st.error(f"An error occurred while setting environment variables: {e}")

@st.cache_resource
def load_chain():
    chain = get_chain()
    return chain

def main():
    # Streamlit UI elements
    st.title("Medical Consultation AI")
    st.write("Ask your medical questions and receive expert advice.")

    # Button to set environment variables
    if st.button("Set Environment Variables"):
        export_environment_variables()

    user_input = st.text_area("Enter your symptoms or concerns:")

    if st.button("Get Advice"):
        if user_input.strip() == "":
            st.error("Please enter your symptoms or concerns.")
        else:
            with st.spinner("Processing..."):
                chain = load_chain()
                try:
                    # Call invoke on rag_chain with user input
                    answer = chain.invoke({"input": user_input})  # Adjust this to match how you want to invoke it
                    st.markdown(answer["answer"])
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
