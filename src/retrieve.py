import os
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)


def load_embeddings(law_name: str, save_dir: str = "embeddings") -> Tuple[List[Dict], np.ndarray]:
    """Load previously saved embeddings and data"""
    emb_path = os.path.join(save_dir, f"{law_name}_embeddings.npy")
    data_path = os.path.join(save_dir, f"{law_name}_data.pkl")
    
    if not os.path.exists(emb_path) or not os.path.exists(data_path):
        raise FileNotFoundError(f"Embeddings for '{law_name}' not found. Please run ingestion first.")
    
    embeddings = np.load(emb_path)
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    return data, embeddings


def search(
    query: str,
    embeddings: np.ndarray,
    data: List[Dict[str, Any]],
    model: SentenceTransformer,
    top_k: int = 5
) -> List[Tuple[Dict[str, Any], float]]:
    """Search in the given embeddings"""
    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    similarities = np.dot(embeddings, query_embedding)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append((data[idx], float(similarities[idx])))
    
    return results