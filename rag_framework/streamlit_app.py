import os
import streamlit as st
from datetime import datetime

from rag_framework.config import Config
from rag_framework.core.embeddings import Embedder
from rag_framework.core.vector_store import VectorStore
from rag_framework.core.retriever import Retriever
from rag_framework.core.chain import RAGChain
from rag_framework.core.llm import LLMClient
from rag_framework.core.evaluator import Evaluator

# Config
cfg = Config()

# Core components
embedder = Embedder(model_name=cfg.embedding_model, device=cfg.embedding_device)
vectorstore = VectorStore(persist_dir=cfg.chroma_dir, collection_name=cfg.collection_name)
retriever = Retriever(vector_store=vectorstore, embedder=embedder, k=cfg.top_k_docs, relevance_bias=0.0)
llm_client = LLMClient(api_key=cfg.groq_api_key, model=cfg.llm_model)
chain = RAGChain(llm_client=llm_client, retriever=retriever, system_prompt=cfg.system_prompt)

# Optional evaluator
evaluator = Evaluator(
    tracking_uri=cfg.mlflow_tracking_uri,
    experiment_name=cfg.mlflow_experiment
) if cfg.app_mode == "eval" else None

# UI
st.title(cfg.streamlit_app)
query = st.text_input("Ask a question:")

# Add text docs
st.subheader("Add new text to knowledge base")
new_text = st.text_area("Enter text to store:")

if st.button("Add to Vector Store"):
    if new_text.strip():
        vectors = embedder.embed_texts([new_text])
        doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        vectorstore.add_texts(ids=[doc_id], texts=[new_text], embeddings=vectors, metadatas=[{"source": "ui"}])

        save_dir = cfg.storage_dir
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"{doc_id}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(new_text)
        st.success(f"Stored and saved as {doc_id}")
    else:
        st.warning("Please enter some text before adding.")

# Run RAG
if st.button("Run RAG"):
    if query.strip():
        answer = chain.run(query)
        st.subheader("Answer")
        st.write(answer)

        if evaluator:
            contexts = [d["content"] for d in retriever.retrieve(query)]
            result = evaluator.evaluate(query=query, context_docs=contexts, answer=answer)
            st.subheader("Evaluation Scores")
            st.json(result["scores"])
            st.subheader("Qualitative Feedback")
            st.write(result["qualitative"])
    else:
        st.warning("Please enter a question.")
