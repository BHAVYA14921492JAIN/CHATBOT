import os
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import time

# ---------------- Streamlit Config ----------------
st.set_page_config(page_title="vComChat - vCommission Assistant",
                   page_icon="🤖", layout="wide")

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
/* Simple plain background */
.stApp {
    background-color: #f5f5f5; /* light grey background */
    font-family: "Poppins", sans-serif;
}

/* Title styling */
h1 {
    font-size: 50px !important;
    text-align: center;
    color: #1f2937; /* dark grey */
}

/* Chat bubbles */
.user-bubble {
    background-color: #1e40af;
    color: white;
    padding: 12px;
    border-radius: 15px;
    margin: 5px;
    max-width: 70%;
}
.assistant-bubble {
    background-color: #10b981;
    color: white;
    padding: 12px;
    border-radius: 15px;
    margin: 5px;
    max-width: 70%;
}
</style>
""", unsafe_allow_html=True)

# ---------------- OpenAI Client ----------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ---------------- Paths ----------------
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# ---------------- Load Knowledge Base ----------------
faq_data = []
with open(os.path.join(BASE_PATH, "faq.txt"), "r", encoding="utf-8") as f:
    content = f.read().split("\n\n")
    for block in content:
        if block.strip():
            q = block.split("\n")[0].replace("Q", "").strip(": ")
            a = "\n".join(block.split("\n")[1:]).replace("A:", "").strip()
            faq_data.append({"text": f"FAQ\nQ: {q}\nA: {a}", "type": "faq"})

policies = []
with open(os.path.join(BASE_PATH, "company policies.txt"), "r", encoding="utf-8") as f:
    policies = f.read().split("\n\n")

policy_data = [{"text": p, "type": "policy"} for p in policies if p.strip()]
knowledge_base = faq_data + policy_data
documents = [item["text"] for item in knowledge_base]

# ---------------- Embeddings & FAISS Index ----------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(documents)
doc_embeddings = np.array(doc_embeddings).astype("float32")

index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

# ---------------- Chatbot Function ----------------
def chatbot(query: str) -> str:
    greetings = ["hi", "hello", "hey", "hii", "heyy", "good morning", "good evening"]
    if query.lower().strip() in greetings:
        return "Hello 👋, I’m vComChat! How can I help you today?"

    query_vec = embedder.encode([query]).astype("float32")
    D, I = index.search(query_vec, k=3)
    retrieved_docs = [documents[i] for i in I[0] if i < len(documents)]
    context = "\n\n".join(retrieved_docs)

    if not context.strip():
        return ("I couldn’t find exact details in company docs. "
                "Could you clarify your query, or contact support@vcommission.com?")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", 
             "content": "You are vComChat, a professional and helpful assistant for vCommission affiliates."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

# ---------------- Streamlit UI ----------------
st.title("🤖 vComChat")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='assistant-bubble'>{msg['content']}</div>", unsafe_allow_html=True)

# Chat input
if query := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": query})
    st.markdown(f"<div class='user-bubble'>{query}</div>", unsafe_allow_html=True)

    with st.chat_message("assistant"):
        with st.spinner("✍️ Generating an answer..."):
            answer = chatbot(query)
            typed_answer = ""
            placeholder = st.empty()
            for char in answer:
                typed_answer += char
                placeholder.markdown(f"<div class='assistant-bubble'>{typed_answer}▌</div>", unsafe_allow_html=True)
                time.sleep(0.02)
            placeholder.markdown(f"<div class='assistant-bubble'>{typed_answer}</div>", unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer})
