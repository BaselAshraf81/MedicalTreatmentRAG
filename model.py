import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.agents.agent_types import AgentType
from langchain.agents import AgentExecutor, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool, StructuredTool, tool
import gc
import os
# Save the Hugging Face token
from huggingface_hub.hf_api import HfFolder
HfFolder.save_token('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')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Set environment variable for Google API Key
os.environ["GOOGLE_API_KEY"] = 'AIzaSyBadUb2oZd7KjS8eY6XH8-AbMhO48nEs0g'

# PDF paths
diagnostic_path = '/content/Dataset.pdf'
doctors_recom = '/content/final_organized_doctors_list.pdf'

# Extract text from PDFs
pdf_diagnostic = extract_text_from_pdf(diagnostic_path)
pdf_doctors = extract_text_from_pdf(doctors_recom)

# Text preprocessing functions
def preprocess_text(text):
    processed_text = re.sub(r'\n•\n', ' • ', text)
    processed_text = processed_text.strip()
    processed_text = re.sub(r'\s+', ' ', processed_text)
    return processed_text

def split_sections(text):
    sections = re.split(r'\n([A-Za-z &]+)\n', text)
    sections = [section.strip() for section in sections if section.strip()]
    formatted_text = {}
    for i in range(0, len(sections), 2):
        section_name = sections[i]
        section_content = sections[i+1] if i+1 < len(sections) else ""
        formatted_text[section_name] = section_content.split(' • ')
    return formatted_text

def format_text(formatted_text):
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

# Process extracted text
diagnostic_text = process_text(pdf_diagnostic)
doctors_text = process_text(pdf_doctors)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
chunks = text_splitter.split_text(diagnostic_text)
chunkkk = text_splitter.split_text(doctors_text)
documents = [Document(page_content=chunk) for chunk in chunks]
documents2 = [Document(page_content=chunk) for chunk in chunkkk]

# Initialize embeddings model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Vector store initialization
vector_store = Chroma.from_documents(documents, embeddings)
vector_store.add_documents(documents2)
retriever = vector_store.as_retriever()

# Language model initialization
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.4,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# System prompt definition
system_prompt = """
You are a highly skilled and experienced medical doctor specializing in respiratory diseases, heart diseases, brain disorders, and bone fractures...
"""

# Create prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create chains
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Sample query for testing
query = " I’ve been experiencing some unusual movements and stiffness in my body lately. Can you tell me more about the symptoms I should be looking out for?"
answer = rag_chain.invoke({"input": query})

# Output answer
print(answer["answer"])