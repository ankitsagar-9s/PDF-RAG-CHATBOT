# RAG PDF Chatbot

A **Retrieval-Augmented Generation (RAG) chatbot** that lets you ask questions about the content of a PDF. It leverages **FAISS** for semantic search, **LangChain** for document loading and embeddings, and **Ollama** as the LLM to generate context-aware answers.

Built with **Streamlit**, it provides an interactive web interface for querying PDFs quickly and easily.

---

## Features

- **PDF Semantic Search** – Search your PDFs efficiently using vector embeddings.  
- **Chunked Document Handling** – Automatically splits PDFs into smaller chunks for more accurate semantic retrieval.  
- **FAISS Vector Store** – Enables fast and efficient similarity search over document embeddings.  
- **Ollama LLM Integration** – Generates detailed answers based on the retrieved context.  
- **Interactive Streamlit Interface** – User-friendly web interface for querying PDFs directly.  

---

## Technologies Used

- **Streamlit** – Frontend framework for creating the interactive web interface.  
- **LangChain** – Handles document loading, splitting, and embeddings.  
- **FAISS** – Vector database used for semantic search.  
- **HuggingFace Embeddings** – Converts text chunks into vector representations.  
- **Ollama** – LLM used to generate answers based on retrieved context.  

---

## How It Works

1. **Load PDF** – The PDF is loaded into the system using `PyPDFLoader`.  
2. **Split into Chunks** – Text is split into smaller, overlapping chunks using `RecursiveCharacterTextSplitter`.  
3. **Create Embeddings** – Each chunk is converted into a vector embedding using HuggingFace embeddings.  
4. **Build FAISS Index** – The embeddings are stored in a FAISS vector store for fast semantic search.  
5. **Query the Index** – User queries are matched to the top `K` most relevant chunks.  
6. **Generate Answer** – Ollama LLM generates a detailed, context-aware answer using the retrieved content.  
