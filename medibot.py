import os
import sys
import asyncio
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.vectorstores import FAISS

DB_FAISS_PATH = "vectorstore/db_faiss"
MODEL_NAME = "microsoft/BioGPT"

# 1) Load FAISS Vectorstore
@st.cache_resource(show_spinner=False)
def get_vectorstore():
    from langchain_huggingface import HuggingFaceEmbeddings
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(DB_FAISS_PATH, embed, allow_dangerous_deserialization=True)

# 2) Load the BioGPT Model Locally
@st.cache_resource(show_spinner=False)
def load_local_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,  # Use -1 for CPU, 0 for GPU if available
        max_new_tokens=150,  # Limit output length
        temperature=0.7,  # Control randomness
        top_p=0.9,  # Nucleus sampling for diversity
        repetition_penalty=1.2,  # Penalize repetitive responses
    )

# 3) Build Dynamic Prompt
def build_prompt(context, question):
    return f"""
Use the information provided in the context to answer the user's question as clearly and directly as possible.
If you don’t know the answer, simply say "I don’t know." Avoid making up information or guessing.
Do not include any information outside the given context.

Context:
{context}

Question:
{question}

Answer:
"""

# 4) Streamlit Chat Application
def main():
    st.set_page_config(page_title="MediAssist AI", page_icon="🩺", layout="wide")
    
    db = get_vectorstore()
    llm = load_local_llm()

    st.markdown("# MediAssist AI")
    st.markdown("#### Your AI-Powered Medical Information Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm MediAssist AI powered by BioGPT. How can I assist you today?"}]

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user", avatar="👤").markdown(msg["content"])
        else:
            st.chat_message("assistant", avatar="🩺").markdown(msg["content"])

    user_input = st.chat_input("Ask your medical question here...")
    if user_input:
        st.chat_message("user", avatar="👤").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant", avatar="🩺"):
            with st.spinner("Analyzing your question..."):
                try:
                    # Retrieve context from FAISS
                    retrieved_docs = db.as_retriever(search_kwargs={"k": 3}).invoke(user_input)
                    context = "\n".join([doc.page_content for doc in retrieved_docs])
                    if not context:
                        context = "I could not find relevant information for your query. Let me still try to help you."

                    # Build prompt dynamically
                    prompt = build_prompt(context, user_input)

                    # Generate response
                    generated_responses = llm(prompt, max_length=150, return_full_text=False)
                    response = generated_responses[0]["generated_text"].strip() if generated_responses else None
                    if not response:
                        response = "I'm sorry, I couldn't find a clear answer. Please consult a medical professional for personalized guidance."

                    # Display response
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    error_msg = f"I'm sorry, I encountered an error: {str(e)}. Please try again."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
