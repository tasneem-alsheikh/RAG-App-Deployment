import os
os.system("pip install faiss-cpu==1.7.4")
import faiss

import streamlit as st
import numpy as np
import faiss
import pickle
from mistralai import Mistral

# Load stored policy chunks and embeddings
# Assume each chunk is a dictionary with keys "policy" and "text"
with open("udst_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# The embeddings array should align with the chunks list
with open("udst_embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

# Mistral API Key
api_key = "uPgaqejCGJ8ZU6Oe0uDmUQl1jzcFtUAv"

def get_text_embedding(text):
    """Generate embedding for the input text using Mistral API."""
    client = Mistral(api_key=api_key)
    embeddings_batch_response = client.embeddings.create(model="mistral-embed", inputs=[text])
    return np.array(embeddings_batch_response.data[0].embedding)

def query_rag(question, selected_policies):
    """Retrieve relevant chunks (from selected policies) and generate a response using Mistral API."""
    if not selected_policies:
        return "⚠️ Please select at least one policy."

    # Filter indices for chunks whose policy is in the selected policies
    filtered_indices = [i for i, chunk in enumerate(chunks) if chunk['policy'] in selected_policies]
    if not filtered_indices:
        return "⚠️ No available data for the selected policies."

    # Filter chunks and embeddings based on selection
    filtered_chunks = [chunks[i] for i in filtered_indices]
    filtered_embeddings = embeddings[filtered_indices]

    # Build a temporary FAISS index on the filtered embeddings
    d = filtered_embeddings.shape[1]
    temp_index = faiss.IndexFlatL2(d)
    temp_index.add(filtered_embeddings)

    # Get embedding for the question
    question_embedding = get_text_embedding(question).reshape(1, -1)

    # Retrieve top 3 relevant chunks from filtered index
    D, I = temp_index.search(question_embedding, k=3)
    retrieved_chunks = [filtered_chunks[i] for i in I[0] if i < len(filtered_chunks)]

    # Check if retrieved chunks have valid text
    if not retrieved_chunks or all(chunk['text'].strip() == "" for chunk in retrieved_chunks):
        return "❌ No relevant information found for your query. Try rephrasing or selecting different policies."

    # Create a context string using the retrieved chunk texts
    formatted_context = "\n\n".join(chunk['text'] for chunk in retrieved_chunks)

    prompt = f"""
    Context information is below.
    ---------------------
    {formatted_context}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {question}
    Answer:
    """

    client = Mistral(api_key=api_key)
    response = client.chat.complete(
        model="mistral-large-latest", 
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ---------- Streamlit UI ----------

# Sidebar
st.sidebar.image("https://www.udst.edu.qa/sites/default/files/2024-04/UDST%20LOGO.jpg", width=250)
st.sidebar.title("📜 Policy Selection")

policy_names = [
    "Student Conduct Policy",
    "Academic Schedule Policy",
    "Sport and Wellness Facilities Policy",
    "Graduate Admissions Policy",
    "Use Library Space Policy",
    "International Student Procedure",
    "Registration Procedure",
    "Scholarship and Financial Assistance",
    "Library Study Room Booking Procedure",
    "Graduate Final Grade Procedure"
]
selected_policies = st.sidebar.multiselect("Select policies to include:", policy_names, default=policy_names)

# Main App
st.title("🎓 UDST Policy Chatbot")
st.write("Welcome to the **UDST Policy Chatbot**! Select policies from the sidebar and ask questions about them.")

st.markdown("""
<style>
    .big-font { font-size:20px !important; }
    .stTextInput>div>div>input { font-size: 18px; }
    .stTextArea textarea { font-size: 16px; }
</style>
""", unsafe_allow_html=True)

# Query input with larger font
query = st.text_input("💬 Enter your question:", placeholder="e.g., What are the rules for using sports facilities?")

# Button to get response
if st.button("🔍 Ask"):
    if query:
        with st.spinner("Thinking... 💭"):
            response = query_rag(query, selected_policies)
        st.markdown("### 🤖 Answer:")
        st.markdown(f"✅ **{response}**", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a question.")
