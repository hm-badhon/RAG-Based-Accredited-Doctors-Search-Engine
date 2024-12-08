# RAG-Based-Accredited-Doctors-Search-Engine

This project is a **Retrieval-Augmented Generation (RAG)** application built using the `gemini-1.5-pro` model. The system allows users to query a PDF containing a list of accredited physicians and retrieve concise, relevant answers. The system responds strictly to questions based on the PDF content and does not generate answers to queries outside its scope.

---

## Features

- **PDF Integration**: Uses a provided PDF document containing details of physicians to answer queries.
- **RAG Architecture**: Combines document retrieval with LLM-based generation for precise responses.
- **Pre-trained LLM**: Powered by the Gemini-1.5-Pro model for high-quality language understanding.
- **Streamlit Interface**: A user-friendly interface for querying the system.

---

## Getting Started

### Prerequisites
- Python 3.9+
- Libraries:
  - `streamlit`
  - `langchain`
  - `chromadb`
  - `langchain_google_genai`
  - `dotenv`

Install the required libraries using:
```bash
pip install -r requirements.txt
```

### Running the Application
1. Clone the repository:
   ```bash
   git clone https://github.com/Taha533/RAG-Based-Accredited-Doctors-Search-Engine.git
   cd rag-physicians-directory
   ```
2. Add the PDF file (`doctors_list.pdf`) to the root directory.
3. Create a `.env` file with your API keys and credentials.
4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

5. Access the application in your web browser

---

## Usage
1. Upload the **`doctors_list.pdf`** (already pre-integrated).
2. Enter a query related to the document, such as:
   - "List of doctors in Dhaka"
   - "Physicians available in Chattogram"
3. The system retrieves and presents relevant answers.

### Example Queries
- **Input**: "Provide the list of doctors from Dhaka."
- **Output**: A concise, formatted list of doctors and their contact details for Dhaka.

---

## System Workflow

1. **Document Loader**: The PDF file is processed and split into smaller chunks using `RecursiveCharacterTextSplitter`.
2. **Vector Store**: Chroma stores vector embeddings of document chunks for similarity-based retrieval.
3. **Retriever**: Uses a similarity search to fetch relevant chunks of text based on user queries.
4. **LLM Integration**: Gemini-1.5-Pro generates answers based on the retrieved content.
5. **Query Handling**: Answers are strictly derived from the provided document; irrelevant questions are met with "I don't know."

---

## Project Structure

```
rag-physicians-directory/
├── qa_gemini_pro.py                  # Main application script
├── doctors_list.pdf        # PDF document with physicians' details
├── requirements.txt        # Required Python libraries
├── .env                    # API keys and configuration
└── README.md               # Project documentation
```

---

## Limitations
- The system only answers queries directly related to the PDF content.
- It cannot generate responses outside the provided context.

---
