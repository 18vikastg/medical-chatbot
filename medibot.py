import os
import sys
import asyncio
import streamlit as st
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline

# ── 1) Windows asyncio fix ───────────────────────────────────────────────────
if sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

DB_FAISS_PATH = "vectorstore/db_faiss"
MODEL_NAME = "google/flan-t5-large"

# ── 2) Load FAISS Vectorstore ────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_vectorstore():
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(DB_FAISS_PATH, embed, allow_dangerous_deserialization=True)

# ── 3) Build Prompt Template ─────────────────────────────────────────────────
def build_prompt():
    return PromptTemplate(
        template="""
You are a helpful and knowledgeable medical assistant. Use the provided context
to answer the user's question clearly and in detail preferebly in 5-6 lines.

Context:
{context}

Question:
{question}

Answer in full sentences, provide examples if helpful.
""",
        input_variables=["context", "question"]
    )

# ── 4) Load the model locally ─────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_local_llm():
    with st.spinner("Loading medical knowledge base. Please wait..."):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1,           # -1 = CPU; use 0 for GPU
            max_new_tokens=512,  # longer answers
            do_sample=True,      # enable sampling
            top_p=0.9,           # nucleus sampling
            top_k=50,            # restrict to top 50 tokens
            num_beams=5          # beam search
        )
        return HuggingFacePipeline(pipeline=pipe)

# ── 5) Streamlit UI + QA loop ────────────────────────────────────────────────
def main():
    # Set page config
    st.set_page_config(
        page_title="MediAssist AI",
        page_icon="🩺",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .main {
        background-color: #f5f9ff;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .st-emotion-cache-16txtl3 h1 {
        color: #2c3e50;
        font-family: 'Arial', sans-serif;
    }
    .disclaimer {
        font-size: 0.8em;
        color: #777;
        border-top: 1px solid #ddd;
        padding-top: 10px;
        margin-top: 20px;
    }
    .chat-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    .user-bubble .st-emotion-cache-jzx5jz {
        background-color: #e3f2fd !important;
    }
    .assistant-bubble .st-emotion-cache-jzx5jz {
        background-color: #e8f5e9 !important;
    }
    .copyright {
        text-align: center;
        margin-top: 20px;
        font-size: 0.8em;
        color: #666;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header with logo and title
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown("# 🩺")
    with col2:
        st.markdown("# MediAssist AI")
        st.markdown("### Your AI-Powered Medical Information Assistant")

    # Info and disclaimer
    with st.expander("ℹ️ About this service"):
        st.markdown("""
        **MediAssist AI** uses advanced natural language processing to provide medical information based on trusted sources.
        
        **Important Notice:**
        - This AI assistant is for informational purposes only
        - Not a substitute for professional medical advice
        - Always consult with qualified healthcare providers for diagnosis and treatment
        - In case of emergency, call your local emergency number immediately
        
        **Powered by**: Flan-T5-Large model with medical knowledge retrieval system
        """)

    # Chat container
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        welcome_msg = {
            "role": "assistant",
            "content": "👋 Hello! I'm MediAssist AI, your medical information assistant. How can I help you today? You can ask me questions about medical conditions, treatments, or general health information."
        }
        st.session_state.messages.append(welcome_msg)

    # Display chat history
    for i, msg in enumerate(st.session_state.messages):
        class_name = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🩺"):
            st.markdown(f"<div class='{class_name}'>{msg['content']}</div>", unsafe_allow_html=True)

    # User input
    user_q = st.chat_input("Ask your medical question here...")
    
    if user_q:
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_q)
        st.session_state.messages.append({"role": "user", "content": user_q})

        # Generate response with progress indicator
        with st.chat_message("assistant", avatar="🩺"):
            with st.spinner("Analyzing medical data..."):
                try:
                    db = get_vectorstore()
                    qa = RetrievalQA.from_chain_type(
                        llm=load_local_llm(),
                        chain_type="stuff",
                        retriever=db.as_retriever(search_kwargs={"k": 3}),
                        chain_type_kwargs={"prompt": build_prompt()}
                    )
                    res = qa.invoke({"query": user_q})
                    answer = res["result"]
                    
                    # Format response
                    current_time = datetime.now().strftime("%H:%M")
                    reply = f"{answer}\n"
                    
                    # Show response
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    
                except Exception as e:
                    error_msg = f"I apologize, but I encountered an error while processing your question. Please try again or rephrase your query. Technical details: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    st.markdown("</div>", unsafe_allow_html=True)

    # Sidebar with additional info
    with st.sidebar:
        st.subheader("MediAssist AI")
        
        st.markdown("#### Features")
        st.markdown("✅ Medical information lookup")
        st.markdown("✅ Evidence-based responses")
        st.markdown("✅ Privacy-focused (runs locally)")
        st.markdown("✅ No internet required after setup")
        
        # Model info
        st.markdown("#### System Status")
        st.success("Medical knowledge base: Active")
        st.info(f"Model: {MODEL_NAME}")
        
        # References section
        st.markdown("#### Resources")
        st.markdown("[CDC Health Information](https://www.cdc.gov/)")
        st.markdown("[WHO Guidelines](https://www.who.int/)")
        st.markdown("[NIH MedlinePlus](https://medlineplus.gov/)")
        
        # Copyright notice
        st.markdown("<div class='copyright'>© 2025 vikastg. All rights reserved.</div>", unsafe_allow_html=True)

    # Footer disclaimer
    st.markdown("<div class='disclaimer'>This AI assistant provides information sourced from medical databases. However, it is not a replacement for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.</div>", unsafe_allow_html=True)
    st.markdown("<div class='copyright'>© 2025 vikastg. All rights reserved. </div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()