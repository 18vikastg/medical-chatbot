import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Set your Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN") or "hf_your_actual_token_here"
HUGGINGFACE_REPO_ID = "microsoft/BioGPT"  # Switch to BioGPT

def load_llm(repo_id):
    return HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.7,  # Adjust temperature for coherence
        huggingfacehub_api_token=HF_TOKEN,
        task="text-generation",  # Suitable for BioGPT
        model_kwargs={"max_new_tokens": 150}  # Limit output length
    )

CUSTOM_PROMPT_TEMPLATE = """
You are a helpful and knowledgeable medical assistant. Use the provided context to answer the user's question clearly and in 5-6 sentences.

Context:
{context}

Question:
{question}

Instructions:
- Be detailed, explain clearly with examples if needed.
- Don’t make up info outside the context.
- Use simple language.
- Encourage consulting a doctor for any critical issues.
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Load your vector store
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA system
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Ask the bot
user_query = input("Write your medical query: ")
response = qa_chain.invoke({"query": user_query})

print("\nRESULT:\n", response["result"])
print("\nSOURCE DOCUMENTS:")
for doc in response["source_documents"]:
    print("-", doc.metadata.get("source", "<unknown>"))
