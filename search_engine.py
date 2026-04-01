import json
import numpy as np
import faiss

# ---> THESE ARE THE NEW IMPORTS WE NEEDED <---
from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential

def load_data_and_build_index(json_filepath):
    print("Loading JSON data...")
    with open(json_filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print("Extracting embeddings...")
    embeddings_list = [row['embedding'] for row in data]
    
    embeddings_matrix = np.array(embeddings_list).astype('float32')
    
    print("Building FAISS Vector Index...")
    dimension = embeddings_matrix.shape[1]
    
    index = faiss.IndexFlatIP(dimension)
    
    faiss.normalize_L2(embeddings_matrix)
    index.add(embeddings_matrix)
    
    print(f"Index built successfully with {index.ntotal} vectors!")
    return data, index

def search_index(query_embedding, data, index, top_k=3):
    query_vector = np.array([query_embedding]).astype('float32')
    faiss.normalize_L2(query_vector)
    
    distances, indices = index.search(query_vector, top_k)
    
    results = []
    for i in range(top_k):
        idx = indices[0][i]
        score = distances[0][i]
        matched_row = data[idx]
        
        results.append({
            "score": float(score),
            "data": matched_row
        })
        
    return results

if __name__ == "__main__":
    # 1. Load your dataset
    my_data, my_index = load_data_and_build_index("stackoverflow_3000_updated.json")
    
    # 2. Setup GitHub Models Client
    # ---> PASTE YOUR TOKEN HERE <---
    GITHUB_TOKEN = "REMOVED_FOR_SECURITY"
    
    print("\nConnecting to GitHub Models API...")
    client = EmbeddingsClient(
        endpoint="https://models.inference.ai.azure.com",
        credential=AzureKeyCredential(GITHUB_TOKEN)
    )
    
    # 3. Simulate a REAL user search!
    user_question = "how to declare array in python"
    print(f'\nUser asked: "{user_question}"')
    print("Converting text to vector embedding...")
    
    response = client.embed(
        input=[user_question],
        model="text-embedding-3-small" 
    )
    
    live_query_vector = response.data[0].embedding
    
    # 4. Search the index with the new vector
    print("\n--- Searching Database ---")
    search_results = search_index(live_query_vector, my_data, my_index, top_k=3)
    
    for rank, result in enumerate(search_results, 1):
        print(f"\nResult {rank} (Match Score: {result['score']:.4f})")
        print(f"Question: {result['data']['question']}")