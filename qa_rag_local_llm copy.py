import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

# Try to import Chroma from langchain_chroma, fallback to langchain_community
try:
    from langchain_chroma import Chroma
    st.info("🛠️ Using langchain_chroma.Chroma for vector store.")
except ImportError:
    from langchain_community.vectorstores import Chroma
    st.warning(
        "⚠️ langchain_chroma not found; using deprecated langchain_community.vectorstores.Chroma. "
        "Install with 'pip install langchain-chroma' to avoid future issues."
    )

# Custom CSS for improved styling
st.markdown("""
    <style>
    .main { background-color: #f5f7fa; padding: 20px; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 5px; }
    .stTextInput>div>input { border-radius: 5px; border: 1px solid #ccc; }
    .stSlider>div { color: #333; }
    .history-card { background-color: #ffffff; padding: 10px; border-radius: 5px; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .st-expander { background-color: #e8ecef; border-radius: 5px; }
    h1 { color: #2c3e50; }
    </style>
""", unsafe_allow_html=True)

load_dotenv()

# Streamlit UI
st.title("🩺 Accredited Doctors Search Engine")
st.markdown("Search for doctors or specialties using our AI-powered tool, powered by local LLMs for privacy and speed.")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    k = st.slider("Number of search results", 1, 20, 10, help="Adjust how many documents are retrieved for each query.")
    temperature = st.slider("LLM Creativity", 0.0, 1.0, 0.0, help="Higher values make responses more creative, lower values more precise.")
    st.markdown("---")
    st.info("Use the admin controls below to manage the PDF data source.")

# Initialize vector store with local LLM embeddings
persist_dir = "chroma_db"
embedding_model = "nomic-embed-text"
try:
    embedding = OllamaEmbeddings(model=embedding_model)
    st.success(f"✅ Initialized embedding model: {embedding_model}")
except Exception as e:
    st.error(
        f"❌ Failed to initialize embedding model '{embedding_model}': {str(e)}. "
        "Ensure Ollama is running and the model is pulled (e.g., 'ollama pull nomic-embed-text')."
    )
    st.stop()

# Check if vector store exists; otherwise, process PDF
if os.path.exists(persist_dir):
    try:
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding)
        st.success("✅ Loaded existing vector store from disk.")
    except Exception as e:
        st.error(f"❌ Failed to load vector store: {str(e)}. Try reprocessing the PDF.")
        st.stop()
else:
    try:
        pdf_path = "/media/nsl47/hdd/AI_project/BAHA/RAG-Based-Accredited-Doctors-Search-Engine/qa_file/doctors_list.pdf"
        if not os.path.exists(pdf_path):
            st.error(
                f"❌ PDF file not found at {pdf_path}. "
                "Please ensure the PDF exists in the 'qa_file' directory."
            )
            st.stop()
        with st.spinner("📄 Processing PDF..."):
            loader = PyPDFLoader(pdf_path)
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.split_documents(data)
            vectorstore = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=persist_dir)
        st.success("✅ Created and saved vector store from PDF.")
    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")
        st.stop()

# Retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

# Local LLM for question-answering
try:
    llm = OllamaLLM(model="llama3.2:1b", temperature=temperature)
    st.success("✅ Initialized LLM: llama3.2:1b")
except Exception as e:
    st.error(
        f"❌ Failed to initialize LLM: {str(e)}. "
        "Ensure Ollama is running and 'llama3.2:1b' is pulled (e.g., 'ollama pull llama3.2:1b')."
    )
    st.stop()

# Prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise.\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Query input
with st.container():
    st.subheader("🔍 Ask About a Doctor or Specialty")
    query = st.text_input(
        "Enter your query (e.g., 'Find a cardiologist' or 'List doctors in New York')",
        help="Type your question here and press 'Search'."
    )
    if st.button("Search", key="search_button"):
        if query and query.strip():
            with st.spinner("🤖 Processing your query..."):
                try:
                    question_answer_chain = create_stuff_documents_chain(llm, prompt)
                    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                    response = rag_chain.invoke({"input": query})
                    formatted_response = "\n\n".join(line.strip() for line in response["answer"].splitlines() if line.strip())
                    if "history" not in st.session_state:
                        st.session_state.history = []
                    st.session_state.history.append({"query": query, "answer": formatted_response})
                    st.markdown(f"**Reply:**\n\n{formatted_response}")
                except Exception as e:
                    st.error(f"❌ Error processing query: {str(e)}")
        else:
            st.warning("⚠️ Please enter a valid query.")

# Query history
with st.expander("📜 Query History", expanded=True):
    if "history" in st.session_state and st.session_state.history:
        if st.button("Clear History", key="clear_history"):
            st.session_state.history = []
            st.rerun()
        for idx, item in enumerate(st.session_state.history):
            with st.container():
                st.markdown(f"<div class='history-card'>**Q{idx+1}:** {item['query']}<br>**A:** {item['answer']}</div>", unsafe_allow_html=True)
                if st.button("Re-run Query", key=f"rerun_{idx}"):
                    query = item["query"]
                    with st.spinner("🤖 Re-running query..."):
                        try:
                            question_answer_chain = create_stuff_documents_chain(llm, prompt)
                            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                            response = rag_chain.invoke({"input": query})
                            formatted_response = "\n\n".join(line.strip() for line in response["answer"].splitlines() if line.strip())
                            st.session_state.history.append({"query": query, "answer": formatted_response})
                            st.markdown(f"**Reply:**\n\n{formatted_response}")
                        except Exception as e:
                            st.error(f"❌ Error re-running query: {str(e)}")
    else:
        st.info("No queries yet. Start searching above!")

# Admin controls
with st.sidebar.expander("🛠️ Admin Controls"):
    uploaded_file = st.file_uploader("Upload New PDF (Optional)", type="pdf", help="Upload a new doctors list PDF (max 200MB).")
    if uploaded_file:
        try:
            with open("temp_doctors_list.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            pdf_path = "temp_doctors_list.pdf"
            st.success("✅ PDF uploaded successfully.")
        except Exception as e:
            st.error(f"❌ Error uploading PDF: {str(e)}")
    if st.button("Reprocess PDF"):
        with st.spinner("📄 Reprocessing PDF..."):
            try:
                if os.path.exists(persist_dir):
                    import shutil
                    shutil.rmtree(persist_dir)
                loader = PyPDFLoader(pdf_path)
                data = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                docs = text_splitter.split_documents(data)
                vectorstore = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=persist_dir)
                st.success("✅ PDF reprocessed and vector store updated.")
            except Exception as e:
                st.error(f"❌ Error reprocessing PDF: {str(e)}")