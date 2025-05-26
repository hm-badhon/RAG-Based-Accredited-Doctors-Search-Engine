import streamlit as st
import time
import logging
import os
from uuid import uuid4
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    st.error("Google API key not found. Please set it in the .env file.")
    st.stop()

# Initialize session state for query history
if "history" not in st.session_state:
    st.session_state.history = []

# Cache vector store creation
@st.cache_resource
def create_vectorstore(_chunk_size=1000):
    try:
        logger.info("Loading and processing PDF...")
        loader = PyPDFLoader("doctors_list.pdf")
        data = loader.load()
        
        # Clean documents (example: remove headers)
        for doc in data:
            doc.page_content = doc.page_content.replace("Header Text", "")
            doc.metadata["source"] = "doctors_list.pdf"
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=_chunk_size)
        docs = text_splitter.split_documents(data)
        logger.info(f"Split into {len(docs)} documents")
        
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
            persist_directory="./chroma_db"
        )
        vectorstore.persist()
        return vectorstore
    except Exception as e:
        logger.error(f"Failed to create vector store: {str(e)}")
        st.error(f"Failed to process PDF: {str(e)}")
        st.stop()

# Streamlit app title
st.title("RAG-Based Accredited Doctors Search Engine")

# Sidebar for configurable parameters and history
with st.sidebar:
    st.subheader("Search Settings")
    chunk_size = st.slider("Chunk size for text splitting", min_value=500, max_value=2000, value=1000, step=100)
    k = st.slider("Number of results to retrieve", min_value=1, max_value=20, value=10)
    
    st.subheader("Query History")
    for item in st.session_state.history:
        st.markdown(f"**Q:** {item['query']}\n\n**A:** {item['answer']}")

# Create vector store
vectorstore = create_vectorstore(chunk_size)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)

# Define system prompt
system_prompt = (
    "You are a medical directory assistant. "
    "Use the provided context to find accredited doctors matching the query. "
    "Include doctor names, specialties, and locations in a concise response (max 3 sentences)."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create RAG chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Chat input
query = st.chat_input("Enter your query (e.g., 'Find a cardiologist in New York')")

# Process query
if query and query.strip():
    try:
        start_time = time.time()
        with st.spinner("Processing your query..."):
            response = rag_chain.invoke({"input": query})
            formatted_response = response["answer"].replace('\n', '\n\n')
            st.markdown(f"**Reply:**\n\n{formatted_response}")
            st.info(f"Processed in {time.time() - start_time:.2f} seconds")
            
            # Store in history
            st.session_state.history.append({"query": query, "answer": response["answer"]})
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        st.error(f"Error processing query: {str(e)}")
elif query:
    st.warning("Please enter a valid query.")