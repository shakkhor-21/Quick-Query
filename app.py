import streamlit as st
import json
import gzip
import numpy as np
import faiss
from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential

# --- CONFIGURATION ---
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
DATA_PATH = "data.json.gz"

# --- UI SETUP ---
st.set_page_config(page_title="DevSearch AI", layout="centered")
st.title("🔍 Semantic Programming Search")
st.write("Ask any programming question and find the best StackOverflow matches.")

# --- LOAD ENGINE (Cached so it only runs once) ---
@st.cache_resource
def load_engine():
    with gzip.open(DATA_PATH, "rt", encoding="utf-8") as f:
        data = json.load(f)
    embeddings = np.array([row['embedding'] for row in data]).astype('float32')
    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return data, index

data, index = load_engine()

# Connect to GitHub Models
client = EmbeddingsClient(
    endpoint="https://models.inference.ai.azure.com", 
    credential=AzureKeyCredential(GITHUB_TOKEN)
)

# --- SEARCH BOX UI ---
query = st.text_input("What are you looking for?", placeholder="e.g., How to declare an array in Python?")

if query:
    with st.spinner("Searching GitHub Models..."):
        # 1. Get Embedding from GitHub
        response = client.embed(input=[query], model="text-embedding-3-small")
        query_vec = np.array([response.data[0].embedding]).astype('float32')
        faiss.normalize_L2(query_vec)
        
        # 2. Search FAISS
        distances, indices = index.search(query_vec, k=3)
        
        # 3. Display Results
        st.subheader("Top Results:")
        for i in range(3):
            res = data[indices[0][i]]
            score = distances[0][i]
            
            # Create a dropdown box for each result
            with st.expander(f"Result {i+1}: {res['question']} (Match: {score:.2%})"):
                st.markdown("**Question Detail:**")
                st.markdown(res['body'], unsafe_allow_html=True)
                st.markdown("---")
                st.markdown("**Top Answer:**")
                st.markdown(res['answer'], unsafe_allow_html=True)