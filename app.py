"""
RAG + Web Search Agent with Gemini API - Beginner-Friendly Implementation
For: Agentic AI Class Assignment

This application demonstrates:
1. PDF processing and text chunking (RAG knowledge base)
2. Embeddings using sentence-transformers
3. Vector storage with FAISS
4. Live web search via Serper API
5. LLM-powered answer generation with Gemini API
"""

import os
import requests
from typing import List
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ============================================================================
# INITIALIZE EXTERNAL SERVICES
# ============================================================================

# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Initialize Gemini client - try new package first, then fallback to old
try:
    import google.genai as genai
    client = genai.Client(api_key=GEMINI_API_KEY)
    GEMINI_CLIENT_TYPE = "genai"
except Exception as e:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_CLIENT_TYPE = "generativeai"
    except Exception as e2:
        st.error("Could not initialize Gemini API. Please check your API key.")
        GEMINI_CLIENT_TYPE = None

# Initialize sentence transformer for embeddings
@st.cache_resource
def load_embedding_model():
    """Load the embedding model once and cache it."""
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# ============================================================================
# HELPER FUNCTIONS - PDF AND TEXT PROCESSING
# ============================================================================

def extract_text_from_pdf(pdf_file) -> str:
    """
    Extract text from uploaded PDF file.
    
    Args:
        pdf_file: Streamlit uploaded file object
        
    Returns:
        str: Extracted text from PDF
    """
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks for RAG.
    
    Args:
        text: Full text to chunk
        chunk_size: Number of characters per chunk
        overlap: Number of overlapping characters between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        # Extract chunk from current position to chunk_size
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move start position, accounting for overlap
        start = end - overlap
    
    return chunks


# ============================================================================
# HELPER FUNCTIONS - EMBEDDINGS AND VECTOR SEARCH
# ============================================================================

def create_faiss_index(chunks: List[str], embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Create a FAISS index from text chunks and their embeddings.
    
    Args:
        chunks: List of text chunks
        embeddings: Numpy array of embeddings for chunks
        
    Returns:
        FAISS index object
    """
    # Ensure embeddings are in float32 format (FAISS requirement)
    embeddings = np.asarray(embeddings).astype('float32')
    
    # Create FAISS index using L2 distance metric
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    return index


def search_faiss_index(
    query: str, 
    index: faiss.IndexFlatL2, 
    chunks: List[str], 
    k: int = 3
) -> List[str]:
    """
    Retrieve top-k most relevant chunks from FAISS index.
    
    Args:
        query: User's question/search query
        index: FAISS index
        chunks: Original text chunks
        k: Number of results to return
        
    Returns:
        List of most relevant text chunks
    """
    # Embed the query
    query_embedding = embedding_model.encode([query], convert_to_tensor=False)
    query_embedding = np.asarray(query_embedding).astype('float32')
    
    # Search the index
    distances, indices = index.search(query_embedding, k)
    
    # Retrieve corresponding chunks
    relevant_chunks = [chunks[i] for i in indices[0]]
    
    return relevant_chunks


# ============================================================================
# HELPER FUNCTIONS - WEB SEARCH
# ============================================================================

def search_web(query: str, num_results: int = 5) -> str:
    """
    Perform a Google search using Serper API.
    
    Args:
        query: Search query
        num_results: Number of results to retrieve
        
    Returns:
        Formatted string of search results
    """
    url = "https://google.serper.dev/search"
    
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "q": query,
        "num": num_results
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        results = response.json()
        
        # Format search results
        formatted_results = "=== WEB SEARCH RESULTS ===\n"
        
        if "organic" in results:
            for i, result in enumerate(results["organic"][:num_results], 1):
                formatted_results += f"\n{i}. {result.get('title', 'No Title')}\n"
                formatted_results += f"   URL: {result.get('link', 'No Link')}\n"
                formatted_results += f"   Snippet: {result.get('snippet', 'No snippet available')}\n"
        
        return formatted_results
        
    except Exception as e:
        return f"Web search error: {str(e)}"


# ============================================================================
# HELPER FUNCTIONS - LLM INTERACTION (GEMINI)
# ============================================================================

def generate_answer(
    question: str, 
    rag_context: str, 
    web_context: str
) -> str:
    """
    Generate answer using Gemini API with combined context.
    
    Args:
        question: User's original question
        rag_context: Relevant chunks from the uploaded PDF
        web_context: Results from web search
        
    Returns:
        Generated answer from Gemini
    """
    if not GEMINI_CLIENT_TYPE:
        return "Gemini API not initialized. Please check your API key."
    
    # Combine all context for the LLM
    combined_context = f"""
Document Context (from uploaded PDF):
{rag_context}

Web Search Context:
{web_context}

Based on the above context, answer this question:
{question}

Provide a comprehensive answer combining both document and web information.
Cite which source (document or web) the information comes from.
"""
    
    try:
        # Try with new google-genai package
        if GEMINI_CLIENT_TYPE == "genai":
            models_to_try = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-pro"]
            
            for model in models_to_try:
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=combined_context
                    )
                    return response.text
                except Exception as model_error:
                    if "429" in str(model_error) or "quota" in str(model_error).lower():
                        return f"API quota exceeded. Please check your Gemini API billing at console.cloud.google.com"
                    continue
        
        # Fallback to legacy google-generativeai package
        else:
            models_to_try = ["gemini-1.5-flash", "gemini-pro"]
            
            for model_name in models_to_try:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(combined_context)
                    return response.text
                except Exception as model_error:
                    if "429" in str(model_error) or "quota" in str(model_error).lower():
                        return f"API quota exceeded. Please check your Gemini API billing at console.cloud.google.com"
                    continue
        
        return "No available Gemini models could generate a response"
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            return f"API quota exceeded. Please check your Gemini API billing and usage limits."
        return f"Error generating answer: {error_msg[:200]}"


# ============================================================================
# STREAMLIT APP CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="RAG + Web Search Agent",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 RAG + Web Search Agent")
st.markdown("Upload a PDF document and ask questions. The agent combines document knowledge with live web search using Gemini API.")

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

# ============================================================================
# MAIN APP LOGIC
# ============================================================================

# Sidebar: PDF Upload and Processing
with st.sidebar:
    st.header("📄 Document Management")
    
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="This will be your RAG knowledge base"
    )
    
    if uploaded_file is not None:
        if st.button("Process PDF", use_container_width=True):
            with st.spinner("Processing PDF..."):
                # Extract text
                pdf_text = extract_text_from_pdf(uploaded_file)
                
                # Split into chunks
                text_chunks = chunk_text(pdf_text)
                
                # Embed chunks
                embeddings = embedding_model.encode(text_chunks, convert_to_tensor=False)
                
                # Create FAISS index
                faiss_index = create_faiss_index(text_chunks, embeddings)
                
                # Store in session state
                st.session_state.faiss_index = faiss_index
                st.session_state.chunks = text_chunks
                st.session_state.pdf_uploaded = True
                
                st.success(f"✅ PDF processed! Created {len(text_chunks)} chunks.")
                st.info(f"Total document length: {len(pdf_text)} characters")
    
    if st.session_state.pdf_uploaded:
        st.success("✅ Document loaded and ready for queries")

# Main area: Question and Answer
st.header("❓ Ask a Question")

if st.session_state.pdf_uploaded:
    user_question = st.text_input(
        "Enter your question:",
        placeholder="What is the main topic of this document?",
        help="Ask anything related to your document or general knowledge"
    )
    
    if st.button("Search & Generate Answer", use_container_width=True):
        if user_question.strip():
            with st.spinner("🔄 Searching document and web..."):
                # Step 1: Retrieve relevant chunks from FAISS
                relevant_chunks = search_faiss_index(
                    user_question,
                    st.session_state.faiss_index,
                    st.session_state.chunks,
                    k=3
                )
                
                rag_context = "\n---\n".join(relevant_chunks)
                
                # Step 2: Perform web search
                web_results = search_web(user_question, num_results=5)
                
                # Step 3: Generate answer with combined context using Gemini
                answer = generate_answer(user_question, rag_context, web_results)
            
            # Display results
            st.subheader("📋 Answer")
            st.write(answer)
            
            # Display sources
            with st.expander("📚 View Source Information"):
                st.subheader("Document Chunks Retrieved:")
                for i, chunk in enumerate(relevant_chunks, 1):
                    st.write(f"**Chunk {i}:**")
                    st.write(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                    st.divider()
                
                st.subheader("Web Search Results:")
                st.write(web_results)
        else:
            st.warning("Please enter a question.")

else:
    st.info("👈 Please upload and process a PDF document first using the sidebar.")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
**How this agent works:**
1. **PDF Processing**: Extract text from uploaded PDF and split into chunks
2. **Embeddings**: Convert chunks into vector embeddings using sentence-transformers
3. **Vector Search**: FAISS stores embeddings and finds relevant chunks based on your question
4. **Web Search**: Serper API performs real-time Google search
5. **LLM Generation**: Gemini combines document + web context to generate comprehensive answers
""")
