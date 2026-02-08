# RAG + Web Search Agent

A simple, beginner-friendly Agentic AI project that combines Retrieval Augmented Generation (RAG) with live web search.

## Project Overview

This is a single-file Streamlit application that demonstrates:
- **RAG (Retrieval Augmented Generation)**: Upload a PDF, extract text, create embeddings, and retrieve relevant chunks
- **Web Search**: Live Google search using Serper API
- **LLM Integration**: Combine document + web context and send to OpenAI for intelligent answers

## Architecture

```
User Input (PDF + Question)
    ↓
[1] PDF Text Extraction → [2] Text Chunking → [3] Embedding (sentence-transformers)
    ↓
[4] FAISS Index Storage (in-memory)
    ↓
Query Processing:
├─ Retrieve relevant chunks from FAISS
├─ Perform web search via Serper API
├─ Combine contexts
└─ Send to OpenAI ChatCompletion
    ↓
Display Answer + Sources in Streamlit
```

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables
- Copy `.env.example` to `.env`
- Add your API keys:
  - `OPENAI_API_KEY`: Get from https://platform.openai.com/api-keys
  - `SERPER_API_KEY`: Get from https://serper.dev

### 3. Run the App
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## How to Use

1. **Upload PDF**: Use the sidebar file uploader to select a PDF document
2. **Process PDF**: Click "Process PDF" button to:
   - Extract text from the PDF
   - Split into overlapping chunks (500 chars with 100 char overlap)
   - Create embeddings using sentence-transformers
   - Build FAISS index for fast similarity search
3. **Ask a Question**: Enter your question in the text input field
4. **Get Answer**: Click "Search & Generate Answer" to:
   - Search FAISS for top 3 relevant document chunks
   - Perform live web search via Serper API
   - Send combined context to OpenAI ChatCompletion
   - Display comprehensive answer citing sources

## Code Structure

### Key Functions (all in `app.py`):

**PDF & Text Processing:**
- `extract_text_from_pdf()` - Extract text from uploaded PDF file
- `chunk_text()` - Split text into overlapping chunks for RAG

**Embeddings & Vector Search:**
- `create_faiss_index()` - Build FAISS index from embeddings
- `search_faiss_index()` - Retrieve top-k most relevant chunks from FAISS

**Web Search & LLM:**
- `search_web()` - Perform Google search using Serper API
- `generate_answer()` - Call OpenAI with combined document + web context

## Technologies Used

- **Streamlit**: Web UI framework
- **OpenAI**: LLM (gpt-3.5-turbo)
- **sentence-transformers**: Embedding model (all-MiniLM-L6-v2)
- **FAISS**: Vector database for similarity search
- **Serper API**: Google Search API
- **PyPDF2**: PDF text extraction

## Key Concepts for Learning

1. **Text Chunking**: Breaking large documents into manageable pieces with overlap
2. **Embeddings**: Converting text to numerical vectors for similarity search
3. **FAISS**: Efficient vector similarity search in-memory
4. **RAG**: Combining retrieved context with LLM for better answers
5. **Web Search Integration**: Augmenting RAG with live web data
6. **Context Combination**: Merging multiple sources for comprehensive answers

## Notes for Classroom Demo

- The embedding model runs locally (no API key needed)
- FAISS index is stored in Streamlit session state (resets on page reload)
- Keep PDF documents under 50 pages for fast processing
- Web search requires active Serper API account
- Model used: gpt-3.5-turbo (set in `call_openai()` function)

## Common Issues

**"API key not found"**: Check your `.env` file has the correct API keys

**"Module not found"**: Make sure all packages from `requirements.txt` are installed

**"Slow PDF processing"**: Large PDFs (100+ pages) may take time. Sentence-transformers will download the model on first run (~50MB)

---

**Created for**: Agentic AI Class Assignment (Beginner Level)
