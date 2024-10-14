import streamlit as st
from model import get_chain



def load_chain():
    chain = get_chain()
    return chain

def main():
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
                    answer = chain.invoke({"input": user_input})
                    st.markdown(answer["answer"])
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
