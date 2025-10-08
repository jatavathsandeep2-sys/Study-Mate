import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import tempfile
import re
from transformers import pipeline
import pandas as pd
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="StudyMate - AI Academic Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #E3F2FD;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .sidebar .sidebar-content {
        background-color: #F5F5F5;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
    .answer-box {
        background-color: #E8F5E9;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .reference-box {
        background-color: #FFF3E0;
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
        font-size: 0.9rem;
    }
    .comparison-box {
        background-color: #F3E5F5;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .similarity-box {
        background-color: #E8F5E9;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .difference-box {
        background-color: #FFEBEE;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
    .feature-tab {
        border-bottom: 2px solid transparent;
        padding: 10px 20px;
        cursor: pointer;
    }
    .feature-tab.active {
        border-bottom: 2px solid #1E88E5;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'chunk_embeddings' not in st.session_state:
    st.session_state.chunk_embeddings = None
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'qa_pipeline' not in st.session_state:
    st.session_state.qa_pipeline = None
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "qa"
if 'saved_comparisons' not in st.session_state:
    st.session_state.saved_comparisons = []

# Title and description
st.markdown('<h1 class="main-header">StudyMate 📚</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="highlight">
    <h3>Your AI-Powered Academic Assistant</h3>
    <p>Upload your academic PDFs (textbooks, lecture notes, research papers) and ask questions in natural language. 
    StudyMate will provide contextual answers extracted directly from your documents.</p>
</div>
""", unsafe_allow_html=True)

# Function to initialize models
@st.cache_resource(show_spinner=False)
def load_models():
    with st.spinner("Loading AI models (this may take a minute)..."):
        # Load embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load question-answering model (small model for faster performance)
        qa_pipeline = pipeline(
            "question-answering", 
            model="distilbert-base-cased-distilled-squad", 
            tokenizer="distilbert-base-cased"
        )
        
        return embedding_model, qa_pipeline

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        doc.close()  # Ensure the document is properly closed
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
    return text

# Function to split text into chunks
def split_text_into_chunks(text):
    # First try splitting by double newlines (paragraphs)
    chunks = re.split(r'\n\s*\n', text)
    chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 100]
    
    # If that doesn't produce good chunks, try splitting by sentences
    if not chunks or len(chunks) < 3:
        # Split by common sentence endings followed by whitespace
        chunks = re.split(r'(?<=[.!?])\s+', text)
        
        # Combine sentences into chunks of reasonable size
        combined_chunks = []
        current_chunk = ""
        for sentence in chunks:
            if len(current_chunk) + len(sentence) < 500:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    combined_chunks.append(current_chunk)
                current_chunk = sentence
        if current_chunk:
            combined_chunks.append(current_chunk)
        
        chunks = [chunk.strip() for chunk in combined_chunks if len(chunk.strip()) > 50]
    
    return chunks

# Function to generate answers from context
def generate_answer(question, context_chunks):
    # Combine context chunks
    context = " ".join([chunk['text'] for chunk in context_chunks])
    
    # Limit context length to avoid model limitations
    max_context_length = 2000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "..."
    
    # Use the QA pipeline to get an answer
    try:
        result = st.session_state.qa_pipeline(question=question, context=context)
        answer = result['answer']
        confidence = result['score']
        
        if confidence < 0.1:  # Low confidence threshold
            answer = "I'm not entirely sure based on the provided documents. The most relevant information I found is: " + answer
        
        return answer
    except:
        # Fallback if QA pipeline fails
        return f"Based on the documents, I found relevant information: {context_chunks[0]['text'][:300]}..."

# Function to compare two terms/concepts
def compare_terms(term1, term2, top_n=5):
    # Find relevant chunks for each term
    term1_embedding = st.session_state.embedding_model.encode([term1])
    term2_embedding = st.session_state.embedding_model.encode([term2])
    
    # Search for similar chunks for term1
    D1, I1 = st.session_state.faiss_index.search(term1_embedding.astype(np.float32), top_n)
    term1_chunks = [st.session_state.chunks[i] for i in I1[0]]
    
    # Search for similar chunks for term2
    D2, I2 = st.session_state.faiss_index.search(term2_embedding.astype(np.float32), top_n)
    term2_chunks = [st.session_state.chunks[i] for i in I2[0]]
    
    # Calculate similarity between the terms based on their context
    similarity_score = np.dot(term1_embedding, term2_embedding.T)[0][0]
    
    # Extract information about each term
    term1_info = extract_term_info(term1, term1_chunks)
    term2_info = extract_term_info(term2, term2_chunks)
    
    # Find similarities and differences
    similarities = find_similarities(term1_info, term2_info)
    differences = find_differences(term1_info, term2_info)
    
    return {
        'term1': term1_info,
        'term2': term2_info,
        'similarity_score': similarity_score,
        'similarities': similarities,
        'differences': differences,
        'term1_chunks': term1_chunks,
        'term2_chunks': term2_chunks
    }

# Function to extract information about a term from relevant chunks
def extract_term_info(term, chunks):
    info = {
        'definition': '',
        'characteristics': [],
        'examples': [],
        'contexts': []
    }
    
    # Look for definitions (sentences that define the term)
    for chunk in chunks:
        sentences = re.split(r'(?<=[.!?])\s+', chunk['text'])
        for sentence in sentences:
            if term.lower() in sentence.lower():
                # Check if this looks like a definition
                if 'is defined as' in sentence.lower() or 'refers to' in sentence.lower() or 'means' in sentence.lower():
                    info['definition'] = sentence
                # Collect characteristics
                if 'characteristic' in sentence.lower() or 'property' in sentence.lower() or 'attribute' in sentence.lower():
                    info['characteristics'].append(sentence)
                # Collect examples
                if 'example' in sentence.lower() or 'for instance' in sentence.lower() or 'such as' in sentence.lower():
                    info['examples'].append(sentence)
                # Collect general contexts
                info['contexts'].append(sentence)
    
    # If no definition found, use the first relevant sentence
    if not info['definition'] and info['contexts']:
        info['definition'] = info['contexts'][0]
    
    return info

# Function to find similarities between two terms
def find_similarities(term1_info, term2_info):
    similarities = []
    
    # Compare characteristics
    term1_chars = set(' '.join(term1_info['characteristics']).lower().split())
    term2_chars = set(' '.join(term2_info['characteristics']).lower().split())
    common_chars = term1_chars.intersection(term2_chars)
    
    if common_chars:
        similarities.append(f"Shared characteristics: {', '.join(common_chars)}")
    
    # Compare contexts
    all_contexts = term1_info['contexts'] + term2_info['contexts']
    common_words = find_common_words(all_contexts)
    
    if common_words:
        similarities.append(f"Common contextual words: {', '.join(common_words[:5])}")
    
    return similarities

# Function to find differences between two terms
def find_differences(term1_info, term2_info):
    differences = []
    
    # Find unique characteristics
    term1_chars = set(' '.join(term1_info['characteristics']).lower().split())
    term2_chars = set(' '.join(term2_info['characteristics']).lower().split())
    unique_term1_chars = term1_chars - term2_chars
    unique_term2_chars = term2_chars - term1_chars
    
    if unique_term1_chars:
        differences.append(f"Unique to {list(term1_info['contexts'])[0].split()[0] if term1_info['contexts'] else 'Term 1'}: {', '.join(unique_term1_chars)}")
    if unique_term2_chars:
        differences.append(f"Unique to {list(term2_info['contexts'])[0].split()[0] if term2_info['contexts'] else 'Term 2'}: {', '.join(unique_term2_chars)}")
    
    return differences

# Function to find common words in contexts
def find_common_words(contexts, top_n=10):
    all_text = ' '.join(contexts).lower()
    words = re.findall(r'\b[a-z]{4,}\b', all_text)
    word_freq = {}
    
    for word in words:
        if word not in ['that', 'with', 'this', 'from', 'which', 'have', 'would', 'there', 'their']:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:top_n]]

# Sidebar for document upload and information
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3612/3612569.png", width=100)
    st.title("Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files", 
        type=['pdf'], 
        accept_multiple_files=True,
        help="Upload one or more academic PDFs to query"
    )
    
    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")
        
        if st.button("Process Documents", key="process_btn"):
            # Load models if not already loaded
            if st.session_state.embedding_model is None or st.session_state.qa_pipeline is None:
                st.session_state.embedding_model, st.session_state.qa_pipeline = load_models()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Extract text from all PDFs
            all_chunks = []
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                progress_bar.progress((i) / len(uploaded_files))
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                
                # Extract text from PDF (with proper file handling)
                text = extract_text_from_pdf(tmp_file_path)
                
                # Clean up temporary file - now safe to delete since PDF is closed
                try:
                    os.unlink(tmp_file_path)
                except PermissionError:
                    # If file is still locked, try again after a short delay
                    import time
                    time.sleep(0.1)
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass  # If still can't delete, just continue
                
                # Skip if no text was extracted
                if not text.strip():
                    st.warning(f"No text could be extracted from {uploaded_file.name}. This file will be skipped.")
                    continue
                
                # Split text into chunks using our improved function
                chunks = split_text_into_chunks(text)
                
                # Add metadata about source document
                for chunk in chunks:
                    all_chunks.append({
                        'text': chunk,
                        'source': uploaded_file.name
                    })
            
            # Check if we have any chunks to process
            if not all_chunks:
                st.error("No text could be extracted from the uploaded documents. Please try different PDF files.")
                st.stop()
            
            status_text.text("Generating embeddings...")
            progress_bar.progress(0.8)
            
            # Store chunks in session state
            st.session_state.chunks = all_chunks
            
            # Generate embeddings for chunks
            chunk_texts = [chunk['text'] for chunk in st.session_state.chunks]
            st.session_state.chunk_embeddings = st.session_state.embedding_model.encode(chunk_texts)
            
            # Create FAISS index
            dimension = st.session_state.chunk_embeddings.shape[1]
            st.session_state.faiss_index = faiss.IndexFlatL2(dimension)
            st.session_state.faiss_index.add(st.session_state.chunk_embeddings.astype(np.float32))
            
            st.session_state.documents_processed = True
            progress_bar.progress(1.0)
            status_text.text("")
            st.success(f"Processing complete! {len(all_chunks)} text chunks extracted.")
    
    st.markdown("---")
    st.markdown("### How to Use")
    st.info("""
    1. Upload one or more PDF documents
    2. Click 'Process Documents'
    3. Ask questions or compare terms
    4. Get answers sourced from your documents
    """)
    
    st.markdown("### Features")
    st.info("""
    ✅ Conversational Q&A from PDFs  
    ✅ Term/Concept Comparison  
    ✅ Accurate text extraction  
    ✅ Semantic search with FAISS  
    ✅ LLM-powered answers  
    ✅ Source references  
    """)

# Feature tabs
col1, col2 = st.columns(2)
with col1:
    if st.button("💬 Q&A Chat", use_container_width=True):
        st.session_state.current_tab = "qa"
with col2:
    if st.button("⚖️ Compare Terms", use_container_width=True):
        st.session_state.current_tab = "compare"

st.markdown("---")

# Main content area
if not st.session_state.documents_processed:
    # Show instructions before processing
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📤 Upload Documents")
        st.write("Upload your textbooks, lecture notes, or research papers in PDF format.")
    
    with col2:
        st.markdown("### 🔍 Ask Questions")
        st.write("Ask natural language questions about the content in your documents.")
    
    with col3:
        st.markdown("### ⚖️ Compare Terms")
        st.write("Find differences and similarities between concepts in your documents.")
    
    st.markdown("---")
    st.markdown("### Supported Document Types")
    
    doc_col1, doc_col2, doc_col3 = st.columns(3)
    with doc_col1:
        st.markdown("**Textbooks**")
        st.write("Comprehensive subject matter with detailed explanations")
    with doc_col2:
        st.markdown("**Lecture Notes**")
        st.write("Course-specific materials from instructors")
    with doc_col3:
        st.markdown("**Research Papers**")
        st.write("Academic papers with specialized content")
        
    st.markdown("---")
    st.markdown("### How It Works")
    st.markdown("""
    1. **Text Extraction**: PyMuPDF extracts text from your PDF documents
    2. **Chunking**: Text is divided into meaningful chunks for processing
    3. **Embeddings**: SentenceTransformers creates numerical representations of text
    4. **Semantic Search**: FAISS finds the most relevant text chunks
    5. **Answer Generation**: A distilled BERT model generates answers from the context
    6. **Term Comparison**: Advanced analysis finds similarities and differences
    """)
else:
    # Show appropriate interface based on selected tab
    if st.session_state.current_tab == "qa":
        # Show chat interface after processing
        st.markdown("### 💬 Ask a question about your documents")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "references" in message and message["references"]:
                    with st.expander("View References"):
                        for ref in message["references"]:
                            st.markdown(f"**Source:** {ref['source']}")
                            st.markdown(f"**Excerpt:** {ref['text'][:200]}...")
        
        # React to user input
        if prompt := st.chat_input("What would you like to know?"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.spinner("Searching documents and generating answer..."):
                # Encode the question
                question_embedding = st.session_state.embedding_model.encode([prompt])
                
                # Search for similar chunks
                D, I = st.session_state.faiss_index.search(question_embedding.astype(np.float32), 5)
                
                # Get the most relevant chunks
                relevant_chunks = [st.session_state.chunks[i] for i in I[0]]
                
                # Generate answer using our QA pipeline
                answer = generate_answer(prompt, relevant_chunks)
                
                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    with st.expander("View References"):
                        for chunk in relevant_chunks:
                            st.markdown(f"**Source:** {chunk['source']}")
                            st.markdown(f"**Excerpt:** {chunk['text'][:200]}...")
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "references": relevant_chunks
                })
    
    elif st.session_state.current_tab == "compare":
        st.markdown("### ⚖️ Compare Terms/Concepts")
        
        col1, col2 = st.columns(2)
        with col1:
            term1 = st.text_input("First term or concept", placeholder="e.g., machine learning")
        with col2:
            term2 = st.text_input("Second term or concept", placeholder="e.g., deep learning")
        
        if st.button("Compare Terms", type="primary") and term1 and term2:
            with st.spinner("Analyzing documents for comparison..."):
                comparison_result = compare_terms(term1, term2)
                
                st.markdown("---")
                st.markdown(f"### Comparison: {term1} vs {term2}")
                
                # Display similarity score
                similarity_percent = min(100, max(0, int(comparison_result['similarity_score'] * 100)))
                st.metric("Semantic Similarity", f"{similarity_percent}%")
                
                # Display definitions
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**{term1}**")
                    if comparison_result['term1']['definition']:
                        st.info(comparison_result['term1']['definition'])
                    else:
                        st.warning("No definition found in documents")
                
                with col2:
                    st.markdown(f"**{term2}**")
                    if comparison_result['term2']['definition']:
                        st.info(comparison_result['term2']['definition'])
                    else:
                        st.warning("No definition found in documents")
                
                # Display similarities
                if comparison_result['similarities']:
                    st.markdown("#### Similarities")
                    for similarity in comparison_result['similarities']:
                        st.markdown(f"""
                        <div class="similarity-box">
                            ✅ {similarity}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No significant similarities found between these terms.")
                
                # Display differences
                if comparison_result['differences']:
                    st.markdown("#### Differences")
                    for difference in comparison_result['differences']:
                        st.markdown(f"""
                        <div class="difference-box">
                            ❌ {difference}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No significant differences found between these terms.")
                
                # Display references
                with st.expander("View Reference Sources"):
                    st.markdown(f"**References for {term1}:**")
                    for chunk in comparison_result['term1_chunks']:
                        st.markdown(f"- **Source:** {chunk['source']}")
                        st.markdown(f"  **Excerpt:** {chunk['text'][:150]}...")
                    
                    st.markdown(f"**References for {term2}:**")
                    for chunk in comparison_result['term2_chunks']:
                        st.markdown(f"- **Source:** {chunk['source']}")
                        st.markdown(f"  **Excerpt:** {chunk['text'][:150]}...")
                
                # Option to save comparison
                if st.button("💾 Save This Comparison"):
                    st.session_state.saved_comparisons.append({
                        "term1": term1,
                        "term2": term2,
                        "result": comparison_result,
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M")
                    })
                    st.success("Comparison saved!")
        
        # Show saved comparisons
        if st.session_state.saved_comparisons:
            st.markdown("---")
            st.markdown("### 💾 Saved Comparisons")
            for i, comp in enumerate(st.session_state.saved_comparisons):
                with st.expander(f"{comp['term1']} vs {comp['term2']} ({comp['date']})"):
                    st.write(f"**Similarity:** {min(100, max(0, int(comp['result']['similarity_score'] * 100)))}%")
                    
                    if st.button("Delete", key=f"delete_comp_{i}"):
                        st.session_state.saved_comparisons.pop(i)
                        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>StudyMate - AI Academic Assistant | Powered by PyMuPDF, FAISS, SentenceTransformers, and Transformers</p>
    <p>No API keys required - 100% open source</p>
</div>
""", unsafe_allow_html=True)