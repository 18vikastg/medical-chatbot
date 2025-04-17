import os
import sys
import asyncio
import streamlit as st

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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
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
    st.title("🩺 Medical Chatbot (Local Flan‑T5‑Large)")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # show chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # get new user query
    user_q = st.chat_input("Ask your medical question here…")
    if not user_q:
        return

    st.chat_message("user").markdown(user_q)
    st.session_state.messages.append({"role":"user","content":user_q})

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

        # Remove any references to source documents since they are not returned
        # docs = res["source_documents"]  # This line should remain removed

        # Reply with the answer only
        reply = f"{answer}\n"
        st.chat_message("assistant").markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

    except Exception as e:
        st.error(f"Error: {e}")

if __name__=="__main__":
    main()
