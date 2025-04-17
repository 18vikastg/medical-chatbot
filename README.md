# 🩺 Medical Chatbot

This is a **Streamlit-based Medical Chatbot** that leverages **Hugging Face LLMs** and **FAISS vector store** to provide context-aware answers to user queries based on medical documents or knowledge provided.  

The chatbot retrieves relevant information from embedded content and uses a language model to generate accurate and contextual responses.

---

## 🚀 Features

- 📚 Contextual Q&A over embedded medical documents.
- 🧠 Uses **HuggingFace**'s `Mistral-7B-Instruct-v0.3` LLM.
- 📦 Vector storage powered by **FAISS**.
- 🌐 Web interface built with **Streamlit**.
- 🔐 Hugging Face API token secured via environment variables.

---

## ⚙️ Setup Guide (Using Pipenv)

### 🔧 Prerequisite

Ensure you have **Pipenv** installed on your system.  
Install it by following the official guide:  
👉 [Install Pipenv](https://pipenv.pypa.io/en/latest/installation.html)

---

### 🛠️ Install Dependencies

Open your terminal and run the following commands:

```bash
pipenv install langchain langchain_community langchain_huggingface faiss-cpu pypdf
pipenv install huggingface_hub
pipenv install streamlit
