import json
import os
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON must be a list of objects")
    return data


def prepare_texts_for_embedding(
    data: List[Dict[str, Any]],
    text_fields: List[str] = None,
    separator: str = " | "
) -> List[str]:
    texts = []
    for item in data:
        if text_fields is None:
            parts = [f"{k}: {v}" for k, v in item.items() if v is not None]
        else:
            parts = [f"{field}: {item.get(field, '')}" for field in text_fields]
        texts.append(separator.join(parts).strip())
    return texts


def create_embeddings(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32
) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings


def save_embeddings(data: List[Dict], embeddings: np.ndarray, law_name: str, save_dir: str = "embeddings"):
    os.makedirs(save_dir, exist_ok=True)
    
    # Save embeddings
    np.save(os.path.join(save_dir, f"{law_name}_embeddings.npy"), embeddings)
    
    # Save original data
    with open(os.path.join(save_dir, f"{law_name}_data.pkl"), "wb") as f:
        pickle.dump(data, f)
    
    print(f"Saved embeddings for {law_name}")


def embed_and_save(
    json_file_path: str,
    law_name: str,
    model_name: str = "all-MiniLM-L6-v2",
    text_fields: List[str] = None,
    save_dir: str = "embeddings"
):
    """Embed a law and save it to disk"""
    print(f"\nProcessing {law_name}...")
    data = load_json_data(json_file_path)
    texts = prepare_texts_for_embedding(data, text_fields=text_fields)
    embeddings = create_embeddings(texts, model_name=model_name)
    
    save_embeddings(data, embeddings, law_name, save_dir)
    return data, embeddings