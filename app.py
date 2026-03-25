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
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
    # Add default first message
    st.session_state.qa_history.append({
        'question': '',
        'answer': 'How can I help you today?',
        'docs': []
    })

st.header("Indecimal AI assistant")

# Show chat history (latest on top)
for entry in reversed(st.session_state.qa_history):
    with st.container():
        st.markdown(f"**You:** {entry['question']}")
        st.success(f"**AI:** {entry['answer']}")
        if entry['docs']:
            st.markdown("**Relevant chunks:**")
            for i, doc in enumerate(entry['docs'], 1):
                with st.expander(f"Chunk {i} - {doc.metadata.get('source', 'Unknown')}"):
                    st.write(doc.page_content)

with st.form(key="chat_form"):
    query = st.text_input("Ask your question")
    submit_button = st.form_submit_button("🚀 Send")

if submit_button:
    if not query:
        st.error("Please enter a question")
    elif not os.path.exists(CHROMA_DB_DIR):
        st.error("Vector store not found. Please process documents first.")
    else:
        with st.spinner("AI is generating response..."):
            try:
                rag_chain, retriever = get_rag_chain()
                answer = rag_chain.invoke(query)
                retrieved_docs = retriever.invoke(query)

                st.session_state.qa_history.append({
                    'question': query,
                    'answer': answer,
                    'docs': retrieved_docs,
                })

                st.rerun()

            except Exception as e:
                st.error(f"❌ Error generating answer: {e}")