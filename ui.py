from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st
pdf_path="data/pdf/diabetes.pdf"
loader=PyPDFLoader(pdf_path)
docs=loader.load()

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20,
    length_function=len,
    separators=["\n\n","\n"," ",""]
)

chunks=text_splitter.split_documents(docs)

EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
FAISS_DIR="faiss_index"
TOP_K=3

embeddings=HuggingFaceEmbeddings(model_name=EMBED_MODEL)

vectorstore=FAISS.from_documents(documents=chunks, embedding=embeddings)
vectorstore.save_local(FAISS_DIR)

reloaded_vs=FAISS.load_local(
    FAISS_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)

st.title(" Semantic Search With FAISS")
try:
    vectorstore=FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    st.success(f"Loaded FAISS index from '{FAISS_DIR}'")
except Exception as e:
    st.error(f"Could not load the FAISS index. Error: {e}")
    st.stop()

QUERY=st.text_input("Enter your query :")
if st.button("Search"):
    if not QUERY.strip():
        st.warning("Please Enter a Query :")
    else:
        results=vectorstore.similarity_search(QUERY,k=TOP_K)
        st.subheader("Top Matches :")

        for i, doc in enumerate(results,1):
            st.markdown(f"### Result {i}")
            st.write(f"**Source:**{doc.metadata.get('source')}")
            st.write(f"**Page :**{doc.metadata.get('page')}")
            st.code(doc.page_content)