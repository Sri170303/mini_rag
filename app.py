import streamlit as st
from utils import fetch_google_drive_md
from rag import process_documents, get_rag_chain
import os
from config import CHROMA_DB_DIR, DATA_DIR

st.set_page_config(page_title="Mini RAG Application", page_icon="📚", layout="wide")

st.title("📚 Mini RAG Application")

# Sidebar for document management
with st.sidebar:
    st.header("📁 Document Management")
    
    # Display list of saved documents
    st.subheader("Saved Documents")
    if os.path.exists(DATA_DIR):
        files = [f for f in os.listdir(DATA_DIR) if f.endswith('.md')]
        if files:
            st.write("Current documents in data folder:")
            for file in files:
                st.write(f"• {file}")
        else:
            st.info("No .md documents found.")
    else:
        st.warning("Data folder does not exist yet.")
    
    st.divider()
    
    # URL input
    url = st.text_input("Paste Google Drive URL for .md file", placeholder="https://drive.google.com/...")
    
    if st.button("📥 Fetch and Save Document", type="primary"):
        if url:
            try:
                filepath = fetch_google_drive_md(url)
                st.success(f"✅ Document saved to {filepath}")
                st.rerun()  # Refresh to show updated list
            except Exception as e:
                st.error(f"❌ Error: {e}")
        else:
            st.error("Please enter a URL")
    
    st.divider()
    
    # Process documents and create vector store
    if st.button("🔄 Process Documents and Create Vector Store", type="secondary"):
        try:
            vectorstore = process_documents()
            st.success("✅ Vector store created and persisted")
        except Exception as e:
            st.error(f"❌ Error processing documents: {e}")

# Main area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("❓ Ask Questions")
    
    # Query
    query = st.text_input("Enter your question about the documents", placeholder="What is the main topic?")
    
    if st.button("🚀 Get Answer", type="primary"):
        if query:
            if os.path.exists(CHROMA_DB_DIR):
                try:
                    rag_chain, retriever = get_rag_chain()
                    
                    # Generate the answer first
                    result = rag_chain.invoke(query)
                    
                    # Check if the answer is a refusal or request for more context
                    if result.strip() == "I cannot answer this question based on the provided documents.":
                        st.subheader("🤖 Answer")
                        st.error(result)
                    elif result.strip() == "I need more context to answer this question. Please provide additional information.":
                        # Retrieve and display context in right column
                        retrieved_docs = retriever.invoke(query)
                        with col2:
                            st.header("📄 Retrieved Context")
                            for i, doc in enumerate(retrieved_docs, 1):
                                with st.expander(f"Chunk {i} - {doc.metadata.get('source', 'Unknown')}"):
                                    st.write(doc.page_content)
                        
                        st.subheader("🤖 Answer")
                        st.warning(result)
                    else:
                        # Retrieve and display context in right column
                        retrieved_docs = retriever.invoke(query)
                        with col2:
                            st.header("📄 Retrieved Context")
                            for i, doc in enumerate(retrieved_docs, 1):
                                with st.expander(f"Chunk {i} - {doc.metadata.get('source', 'Unknown')}"):
                                    st.write(doc.page_content)
                        
                        st.subheader("🤖 Answer")
                        st.success(result)
                except Exception as e:
                    st.error(f"❌ Error generating answer: {e}")
            else:
                st.error("Vector store not found. Please process documents first.")
        else:
            st.error("Please enter a question")

with col2:
    st.header("📄 Retrieved Context")
    st.info("Relevant document chunks will appear here after asking a question.")