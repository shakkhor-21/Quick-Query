import streamlit as st
import json
import gzip # 1. MUST HAVE THIS IMPORT
import numpy as np
import faiss
from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential

# --- CONFIGURATION ---
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
DATA_PATH = "data.json.gz" # 2. MUST BE THE .gz FILE

# --- UI SETUP ---
st.set_page_config(page_title="DevSearch AI", layout="centered")
st.title("🔍 Semantic Programming Search")
st.write("Ask any programming question and find the best StackOverflow matches.")

# --- LOAD ENGINE ---
@st.cache_resource
def load_engine():
    # 3. MUST USE gzip.open AND "rt" 
    with gzip.open(DATA_PATH, "rt", encoding="utf-8") as f:
        data = json.load(f)
    embeddings = np.array([row['embedding'] for row in data]).astype('float32')
    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return data, index