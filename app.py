import streamlit as st
import os
from src.ingest import embed_and_save
from src.retrieve import load_embeddings, get_embedding_model, search

# ====================== CONFIG ======================
MODEL_NAME = "all-MiniLM-L6-v2"
DATA_DIR = "data"
EMB_DIR = "embeddings"
TOP_K = 5

# Available laws
AVAILABLE_LAWS = {
    "IPC": "ipc.json",
    "CrPC": "crpc.json",
    "CPC": "cpc.json"
}

# ====================== PAGE CONFIG ======================
st.set_page_config(page_title="Legal Research RAG Agent", page_icon="⚖️", layout="centered")
st.title(" Legal Research RAG Agent")
st.markdown("Search across IPC, CrPC, CPC and more")

# ====================== SIDEBAR ======================
st.sidebar.header("Settings")

selected_law = st.sidebar.selectbox(
    "Select Law",
    options=["All"] + list(AVAILABLE_LAWS.keys()),
    index=0
)

top_k = st.sidebar.slider("Number of results", 3, 10, 5)

# Button to re-ingest all laws
if st.sidebar.button("🔄 Re-embed All Laws"):
    with st.spinner("Embedding all laws... This may take some time"):
        for law_name, filename in AVAILABLE_LAWS.items():
            json_path = os.path.join(DATA_DIR, filename)
            if os.path.exists(json_path):
                embed_and_save(json_path, law_name.lower(), model_name=MODEL_NAME, save_dir=EMB_DIR)
            else:
                st.sidebar.warning(f"{filename} not found")
    st.sidebar.success("All laws embedded successfully!")
    st.cache_resource.clear()
    st.rerun()

# ====================== LOAD MODEL (once) ======================
@st.cache_resource
def load_model():
    return get_embedding_model(MODEL_NAME)

model = load_model()

# ====================== LOAD SELECTED DATA ======================
@st.cache_resource
def load_selected_data(law: str):
    if law == "All":
        all_data = []
        all_embeddings = []
        
        for law_name in AVAILABLE_LAWS.keys():
            try:
                data, embeddings = load_embeddings(law_name.lower(), save_dir=EMB_DIR)
                all_data.extend(data)
                all_embeddings.append(embeddings)
            except FileNotFoundError:
                st.warning(f"Embeddings for {law_name} not found. Please re-embed.")
                continue
        
        if not all_embeddings:
            return [], None
        
        import numpy as np
        combined_embeddings = np.vstack(all_embeddings)
        return all_data, combined_embeddings
    else:
        data, embeddings = load_embeddings(law.lower(), save_dir=EMB_DIR)
        return data, embeddings


try:
    data, embeddings = load_selected_data(selected_law)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Please click **Re-embed All Laws** in the sidebar first.")
    st.stop()

# ====================== CHAT INTERFACE ======================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input(f"Ask a question about {selected_law}..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching legal documents..."):
            results = search(
                query=prompt,
                embeddings=embeddings,
                data=data,
                model=model,
                top_k=top_k
            )

        if not results:
            response = "No relevant results found."
            st.write(response)
        else:
            response_parts = []
            for i, (record, score) in enumerate(results, 1):
                st.markdown(f"### Result {i} — Score: `{score:.4f}`")
                
                for key, value in record.items():
                    st.markdown(f"**{key}:** {value}")
                
                st.divider()
                response_parts.append(f"**Result {i}** (score: {score:.4f})")

            response = "\n".join(response_parts)

        st.session_state.messages.append({"role": "assistant", "content": response})