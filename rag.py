# rag.py
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from config import DATA_DIR, CHROMA_DB_DIR, EMBEDDING_MODEL, LLM_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, HEADERS_TO_SPLIT_ON

def process_documents():
    """Load, split, and create vector store from documents."""
    loader = DirectoryLoader(DATA_DIR, glob="*.md")
    documents = loader.load()
    
    # Splitting logic
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
    docs = []
    for doc in documents:
        splits = markdown_splitter.split_text(doc.page_content)
        for split in splits:
            new_metadata = {**doc.metadata, **split.metadata}
            new_doc = Document(page_content=split.page_content, metadata=new_metadata)
            docs.append(new_doc)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = text_splitter.split_documents(docs)
    
    # Embeddings
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # Vector store
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DB_DIR)
    vectorstore.persist()
    return vectorstore

def get_rag_chain():
    """Create and return the RAG chain."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever()
    
    llm = ChatOllama(model=LLM_MODEL, temperature=0.7)
    
    prompt = ChatPromptTemplate.from_template("""You are an AI assistant. Answer the following question based only on the provided context. If the context does not contain sufficient information to answer the question, respond with exactly: "I need more context to answer this question. Please provide additional information." If the question is completely unrelated to the documents, respond with exactly: "I cannot answer this question based on the provided documents."

<context>
{context}
</context>

Question: {question}""")
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever