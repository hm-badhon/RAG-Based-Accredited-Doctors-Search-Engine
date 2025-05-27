import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
from datetime import datetime

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

# Custom CSS for chat-style interface
st.markdown("""
    <style>
    .main { background-color: #f5f7fa; padding: 20px; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 5px; }
    .stChatInput>div>input { border-radius: 5px; border: 1px solid #ccc; }
    .stSlider>div { color: #333; }
    .chat-container { max-height: 300px; overflow-y: auto; padding: 10px; background-color: #ffffff; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .user-message { background-color: #d1e7ff; border-radius: 10px; padding: 10px; margin: 10px 0; margin-left: 20%; text-align: right; }
    .assistant-message { background-color: #e6ffed; border-radius: 10px; padding: 10px; margin: 10px 0; margin-right: 20%; text-align: left; }
    .message-timestamp { font-size: 0.8em; color: #666; }
    .st-expander { background-color: #e8ecef; border-radius: 5px; }
    h1 { color: #2c3e50; }
    </style>
""", unsafe_allow_html=True)

load_dotenv()

# Streamlit UI
st.title("🩺 Accredited Doctors Search Engine")
st.markdown("Chat with our AI to find doctors or specialties, powered by local LLMs for privacy and speed.")

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

# Chat input section
with st.container():
    st.subheader("💬 Chat with the Assistant")
    if "latest_query" not in st.session_state:
        st.session_state.latest_query = None
        st.session_state.latest_response = None
        st.session_state.latest_timestamp = None

    query = st.chat_input("Ask about a doctor or specialty (e.g., 'Find a cardiologist')")
    if query and query.strip():
        with st.spinner("🤖 Processing your query..."):
            try:
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                response = rag_chain.invoke({"input": query})
                formatted_response = "\n\n".join(line.strip() for line in response["answer"].splitlines() if line.strip())
                st.session_state.latest_query = query
                st.session_state.latest_response = formatted_response
                st.session_state.latest_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")

    # Display latest conversation
    if st.session_state.latest_query and st.session_state.latest_response:
        with st.container():
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            # User message
            st.markdown(
                f"""
                <div class="user-message">
                    <strong>👤 You:</strong> {st.session_state.latest_query}<br>
                    <span class="message-timestamp">{st.session_state.latest_timestamp}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            # Assistant message
            st.markdown(
                f"""
                <div class="assistant-message">
                    <strong>🩺 Assistant:</strong> {st.session_state.latest_response}<br>
                    <span class="message-timestamp">{st.session_state.latest_timestamp}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Start chatting above to find doctors or specialties!")

# Admin controls
with st.sidebar.expander("🛠️ Admin Controls"):
    uploaded_file = st.file_uploader("Upload New PDF (Optional)", type="pdf", help="Upload a new doctors list PDF (max 200MB).")
    if uploaded_file:
        try:
            if uploaded_file.size > 200 * 1024 * 1024:
                st.error("❌ Uploaded file exceeds 200MB limit.")
                st.stop()
            with open("temp_doctors_list.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            pdf_path = "temp_doctors_list.pdf"
            st.success("✅ PDF uploaded successfully.")
        except Exception as e:
            st.error(f"❌ Error uploading PDF: {str(e)}")
    if st.button("Reprocess PDF", help="Rebuild the vector store with the current PDF"):
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