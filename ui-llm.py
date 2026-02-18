import os
import streamlit as st
import ollama

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# Config

PDF_PATH = "data/pdf/diabetes.pdf"
FAISS_DIR = "faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemma3:1b"   # Make sure this model is pulled
TOP_K = 3

# Streamlit Title

st.title("Semantic Search with FAISS + Ollama")

# Loading Embedding Model

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)


# Creating FAISS index if not exists

if not os.path.exists(FAISS_DIR):
    st.info("Creating FAISS index... (first time only)")

    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=20,
    )

    chunks = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_DIR)

    st.success("FAISS index created and saved!")


# Loading FAISS index

try:
    vectorstore = FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    st.success(f"Loaded FAISS index from '{FAISS_DIR}'")
except Exception as e:
    st.error(f"Error loading FAISS index: {e}")
    st.stop()

# User Query

query = st.text_input("Enter your query:", key="user_query") 

if st.button("Search", key="search_button"):  

    if not query.strip():
        st.warning("Please enter a query")
        st.stop()

    # Retrieving top documents
    results = vectorstore.similarity_search(query, k=TOP_K)

    st.subheader("Top Matches")
    context_text = ""

    for i, doc in enumerate(results, 1):
        st.markdown(f"### Result {i}")
        st.write(f"**Source:** {doc.metadata.get('source')}")
        st.write(f"**Page:** {doc.metadata.get('page')}")
        st.code(doc.page_content)
        context_text += doc.page_content + "\n"

    #Asking Ollama
    prompt = f"""
You are a helpful AI assistant.
Answer the question using ONLY the context below.
Also explain a bit about the answer.

Context:
{context_text}

Question:
{query}
"""

    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )

        st.subheader("Ollama Answer")
        st.write(response["message"]["content"])

    except Exception as e:
        st.error(f"Ollama error: {e}")
