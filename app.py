import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os

st.title("📄 RAG Prototype: Step 1")

# 1. The Upload Widget
uploaded_file = st.file_uploader("Upload your GTU Notes", type="pdf")

if uploaded_file is not None:
    # 2. Save file temporarily (LangChain needs a real file path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # 3. Load the PDF
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    # 4. Success Message
    st.success(f"File Uploaded! It has {len(pages)} pages.")
    
    # 5. Show a sneak peek of page 1
    st.markdown("### Preview of Page 1:")
    st.info(pages[0].page_content)
    
    # Cleanup: remove the temp file
    os.remove(tmp_path)