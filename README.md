RAG PDF Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that enables semantic search over PDF documents and generates AI-powered answers using the Ollama LLM. The system leverages FAISS for vector-based retrieval and HuggingFace embeddings to convert text into semantic vectors. Users can query a PDF and receive contextual answers with explanations.

Features

Semantic search over PDF documents using FAISS.

Chunking large PDFs for effective retrieval using RecursiveCharacterTextSplitter.

Integration with Ollama for LLM-powered question answering.

Streamlit web interface for interactive queries.

Displays top matching PDF segments along with AI-generated responses.

Architecture

Document Loading
PDFs are loaded using PyPDFLoader from langchain_community.document_loaders.

Text Chunking
Documents are split into smaller chunks (300 characters with 20-character overlap) to improve retrieval performance.

Vector Store

Embeddings: Generated using HuggingFaceEmbeddings (sentence-transformers/all-MiniLM-L6-v2).

Vector Index: Stored in FAISS for fast similarity searches.

Retrieval-Augmented Generation (RAG)

Top-k relevant chunks are retrieved for a user query.

Context is sent to the Ollama LLM (gemma3:1b) to generate a concise and informative answer.

Streamlit Interface
Users can enter queries and view:

Top matching document segments
