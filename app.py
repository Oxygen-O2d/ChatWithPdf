import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# --- Imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Load API Key
load_dotenv()

# 2. Page Config
st.set_page_config(page_title="GTU Notes Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Chat with your GTU Notes")

# 3. Session State Setup
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR: Data Ingestion (The Factory) ---
with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process & Ingest"):
            with st.spinner("Processing PDF..."):
                try:
                    # A. Save temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    # B. Load & Split
                    loader = PyPDFLoader(tmp_path)
                    pages = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_documents(pages)
                    
                    # C. Embed (Local) & Store
                    st.write("🧠 Generating Local Embeddings...")
                    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    
                    vector_store = Chroma.from_documents(chunks, embedding_model)
                    
                    # D. Save to Session
                    st.session_state.vector_store = vector_store
                    st.success(f"✅ Indexed {len(chunks)} chunks!")
                    
                    # Cleanup
                    os.remove(tmp_path)
                    
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")

# --- MAIN AREA: Tabs ---
tab1, tab2 = st.tabs(["💬 Chat", "⚡ Flashcards"])

# --- TAB 1: Chat Interface ---
with tab1:
    st.subheader("Chat with your Notes")
    
    # Display History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask a question about the PDF..."):
        # Show User Message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate Response
        if st.session_state.vector_store is not None:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # 1. Setup LLM (Using a stable version)
                        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
                        
                        # 2. Setup Retriever
                        retriever = st.session_state.vector_store.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 3}
                        )
                        
                        # 3. Prompt Template
                        template = """Answer the question based only on the following context:
                        {context}

                        Question: {question}
                        """
                        prompt_template = ChatPromptTemplate.from_template(template)

                        # 4. LCEL Chain (The Modern Way)
                        def format_docs(docs):
                            return "\n\n".join(doc.page_content for doc in docs)

                        rag_chain = (
                            {"context": retriever | format_docs, "question": RunnablePassthrough()}
                            | prompt_template
                            | llm
                            | StrOutputParser()
                        )
                        
                        # 5. Run
                        answer = rag_chain.invoke(prompt)
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.warning("⚠️ Please upload a PDF in the sidebar first!")

# --- TAB 2: Flashcards ---
with tab2:
    st.subheader("Generate Study Flashcards")
    st.markdown("Enter a topic from your notes, and AI will generate 5 practice cards.")
    
    topic = st.text_input("Enter Topic (e.g., 'Backpropagation', 'OSPF', 'Gradient Descent')")
    
    if st.button("Generate Cards"):
        if st.session_state.vector_store is None:
            st.warning("⚠️ Please upload a PDF first!")
        else:
            with st.spinner(f"Creating flashcards for '{topic}'..."):
                try:
                    # 1. Setup LLM
                    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
                    
                    # 2. Retriever (Broader search for flashcards)
                    retriever = st.session_state.vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 4} 
                    )
                    
                    # 3. Strict Prompt for Formatting
                    flashcard_prompt = """
                    You are a teacher creating study flashcards. 
                    Based ONLY on the following context, create 5 flashcards about the topic: '{topic}'.
                    
                    Format each flashcard exactly like this:
                    Question ||| Answer
                    
                    Do not add numbering, bullet points, or extra text. Just the raw lines.
                    
                    Context:
                    {context}
                    """
                    prompt_template = ChatPromptTemplate.from_template(flashcard_prompt)
                    
                    # 4. Chain
                    def format_docs(docs):
                        return "\n\n".join(doc.page_content for doc in docs)
                        
                    chain = (
                        {"context": retriever | format_docs, "topic": RunnablePassthrough()}
                        | prompt_template
                        | llm
                        | StrOutputParser()
                    )
                    
                    # 5. Run & Parse
                    raw_response = chain.invoke(topic)
                    
                    # Display Cards
                    lines = raw_response.strip().split('\n')
                    valid_cards = [line for line in lines if "|||" in line]
                    
                    if not valid_cards:
                        st.warning("AI couldn't find enough info to make flashcards on this topic.")
                    else:
                        st.success(f"Generated {len(valid_cards)} Flashcards!")
                        for line in valid_cards:
                            parts = line.split("|||")
                            if len(parts) == 2:
                                question = parts[0].strip()
                                answer = parts[1].strip()
                                with st.expander(f"❓ {question}"):
                                    st.write(f"**Answer:** {answer}")
                                    
                except Exception as e:
                    st.error(f"Error generating flashcards: {e}")