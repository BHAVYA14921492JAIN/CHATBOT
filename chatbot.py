import os
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

st.set_page_config(page_title="vCommission Affiliate Chatbot", page_icon="🤖", layout="centered")

# -------- API KEY --------
# Reads your key from Streamlit secrets (set in Settings → Secrets)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# -------- Load FAQ --------
faq_data = []
with open(os.path.join(BASE_PATH, "faq.txt"), "r", encoding="utf-8") as f:
    content = f.read().split("\n\n")
    for block in content:
        if block.strip():
            q = block.split("\n")[0].replace("Q", "").strip(": ")
            a = "\n".join(block.split("\n")[1:]).replace("A:", "").strip()
            faq_data.append({"text": f"FAQ\nQ: {q}\nA: {a}", "type": "faq"})

# -------- Load Policies --------
policies = []
with open(os.path.join(BASE_PATH, "company policies.txt"), "r", encoding="utf-8") as f:
    policies = f.read().split("\n\n")

policy_data = [{"text": p, "type": "policy"} for p in policies if p.strip()]

# -------- Combine into one knowledge base --------
knowledge_base = faq_data + policy_data
documents = [item["text"] for item in knowledge_base]

# -------- Embeddings + FAISS --------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(documents)
doc_embeddings = np.array(doc_embeddings).astype("float32")

index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

# -------- Chatbot Logic --------
def chatbot(query: str) -> str:
    query_vec = embedder.encode([query]).astype("float32")
    D, I = index.search(query_vec, k=3)  # get top 3 results

    retrieved_docs = [documents[i] for i in I[0] if i < len(documents)]
    context = "\n\n".join(retrieved_docs)

    if not context.strip():
        return "The provided context does not contain information about this topic."

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # ✅ correct model
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant for vCommission affiliates and advertisers. "
                    "If the context is relevant but phrased differently, still try to give the best possible answer. "
                    "If nothing is relevant at all, reply: "
                    "I'm not entirely sure about that, but you can reach out to us directly: "
                    "Email: support@vcommission.com"
                )
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0
    )
    return response.choices[0].message.content

# -------- Streamlit UI --------
st.title("🤖 vCommission Chatbot")
st.write("Ask me about company policies, payments, or general info!")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Input box
if query := st.chat_input("Type your question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = chatbot(query)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
