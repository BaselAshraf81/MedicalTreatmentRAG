import streamlit as st
from model import get_chain
import subprocess
import os
os.environ["GOOGLE_API_KEY"] = 'AIzaSyBadUb2oZd7KjS8eY6XH8-AbMhO48nEs0g'

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = 'python'
env = os.environ.copy()
env["GOOGLE_API_KEY"] = 'AIzaSyBadUb2oZd7KjS8eY6XH8-AbMhO48nEs0g'
env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = 'python'

# Now run a command in this modified environment
subprocess.run("echo $GOOGLE_API_KEY; echo $PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", shell=True, env=env)

@st.cache_resource
def load_chain():
    chain = get_chain()
    return chain

def main():
    subprocess.run("export GOOGLE_API_KEY='AIzaSyBadUb2oZd7KjS8eY6XH8-AbMhO48nEs0g'", shell=True, executable="/bin/bash")

    # Now, you can access the variable in your script
    api_key = os.getenv("GOOGLE_API_KEY")
    st.title("Medical Consultation AI")
    st.write("Ask your medical questions and receive expert advice.")

    user_input = st.text_area("Enter your symptoms or concerns:")

    if st.button("Get Advice"):
        if user_input.strip() == "":
            st.error("Please enter your symptoms or concerns.")
        else:
            with st.spinner("Processing..."):
                chain = load_chain()
                try:
                    # Call invoke on rag_chain with user input
                    answer=chain.invoke({"input": user_input}) # Adjust this to match how you want to invoke it
                    st.markdown(answer["answer"])
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
