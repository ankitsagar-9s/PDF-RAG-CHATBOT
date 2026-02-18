# PDF RAG Chatbot

A **Retrieval-Augmented Generation (RAG) chatbot** that allows you to ask questions about a PDF document.  
The chatbot retrieves relevant content from the PDF using **FAISS + HuggingFace embeddings** and generates answers using the **Ollama LLM (`gemma3:1b`)**.

---

## Features

- Ask questions about a PDF document with semantic search.
- Uses FAISS for fast similarity search over PDF chunks.
- Embeds PDF text with `sentence-transformers/all-MiniLM-L6-v2`.
- Generates context-aware answers using Ollama LLM.
- Interactive **Streamlit** interface showing top matching pages.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/username/PDF-RAG-CHATBOT.git
cd PDF-RAG-CHATBOT
