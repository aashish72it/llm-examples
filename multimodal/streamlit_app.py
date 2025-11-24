# ---- Force Streamlit server options BEFORE importing streamlit
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"   # disable hot-reload watcher
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"        # don't rerun on save

import uuid
from datetime import datetime
from pathlib import Path
import traceback
import streamlit as st

# ---- Page config FIRST (best practice)
st.set_page_config(page_title="Multimodal Finance Assistant", page_icon="🤖", layout="wide")

# ---- Debug: confirm server options and track reloads
if "reloads" not in st.session_state:
    st.session_state.reloads = 0
st.session_state.reloads += 1

with st.expander("⚙️ Debug (server options & reloads)", expanded=False):
    st.write("fileWatcherType:", st.get_option("server.fileWatcherType"))
    st.write("runOnSave:", st.get_option("server.runOnSave"))
    st.write("Script reload count:", st.session_state.reloads)

# ---- Imports from codebase (after Streamlit is initialized)
from multimodal.utils.feedback_logger import log_feedback, log_mlflow_feedback
from multimodal.config import Config
from multimodal.core.embeddings import Embedder
from multimodal.core.vector_store import VectorStore
from multimodal.core.retriever import Retriever
from multimodal.core.chain import RAGChain
from multimodal.core.llm import LLMClient
from multimodal.core.evaluator import Evaluator
from multimodal.utils.errors import LLMError, RAGError

# ---- Load config 
try:
    cfg = Config()
except Exception as e:
    st.error(f"Config initialization failed: {e}")
    st.exception(traceback.format_exc())
    st.stop()


for p in [cfg.chroma_dir, cfg.storage_dir, cfg.feedback_dir, cfg.log_dir, cfg.upload_dir]:
    if p:
        Path(p).mkdir(parents=True, exist_ok=True)

# ---- Cache heavy resources so reruns don't rebuild them
@st.cache_resource(show_spinner=False)
def get_embedder():
    return Embedder()

@st.cache_resource(show_spinner=False)
def get_vector_store(persist_dir: str, collection_name: str):
    return VectorStore(persist_dir=persist_dir, collection_name=collection_name)

@st.cache_resource(show_spinner=False)
def get_llm_client(model: str, api_key: str):
    return LLMClient(api_key=api_key, model=model)

@st.cache_resource(show_spinner=False)
def get_chain_and_retriever():
    embedder = get_embedder()
    text_store  = get_vector_store(cfg.chroma_dir, cfg.text_collection)
    image_store = get_vector_store(cfg.chroma_dir, cfg.image_collection)
    audio_store = get_vector_store(cfg.chroma_dir, cfg.audio_collection)
    video_store = get_vector_store(cfg.chroma_dir, cfg.video_collection)

    vector_stores = {"text": text_store, "audio": audio_store, "image": image_store, "video": video_store}
    embedders     = {"text": embedder,   "audio": embedder,   "image": embedder,   "video": embedder}
    k_values      = {"text": cfg.top_k_text, "image": cfg.top_k_image, "audio": cfg.top_k_audio, "video": cfg.top_k_video}

    retriever = Retriever(
        vector_stores=vector_stores,
        embedders=embedders,
        k_values=k_values,
        relevance_bias=0.0
    )
    llm_client = get_llm_client(cfg.llm_model, cfg.groq_api_key)
    chain = RAGChain(llm_client=llm_client, retriever=retriever, system_prompt=cfg.system_prompt_multimodal)
    return chain, retriever, embedder, text_store

try:
    chain, retriever, embedder, text_store = get_chain_and_retriever()
except Exception as e:
    st.error(f"Failed to initialize core components: {e}")
    st.exception(traceback.format_exc())
    st.stop()

# ---- Optional evaluator (guard MLflow URI)
evaluator = None
if cfg.app_mode == "eval":
    try:
        evaluator = Evaluator(
            tracking_uri=cfg.mlflow_tracking_uri,
            experiment_name=cfg.mlflow_experiment
        )
    except Exception as e:
        st.warning(f"Evaluator disabled (MLflow init failed): {e}")

# ---- Session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# ---- UI
st.title(cfg.streamlit_app)
query = st.text_input("Ask a question:")

# ---- Run RAG
if st.button("Run RAG"):
    if query.strip():
        with st.spinner("Running RAG pipeline..."):
            try:
                answer = chain.run(query)
                st.session_state.last_query = query
                st.session_state.last_answer = answer

                st.subheader("Answer")
                st.write(answer)

                # --- EVAL
                if evaluator:
                    raw_results = retriever.retrieve(query)
                    contexts = []

                    for modality, res in raw_results.items():
                        if res and "documents" in res and res["documents"]:
                            contexts.extend(res["documents"][0])

                    result = evaluator.evaluate(query=query, context_docs=contexts, answer=answer)
                    st.subheader("Evaluation Scores")
                    st.json(result.get("scores", {}))
                    st.subheader("Qualitative Feedback")
                    st.write(result.get("qualitative", ""))

            except (LLMError, RAGError) as e:
                st.error(f"Pipeline error: {e}")
                st.exception(traceback.format_exc())
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                st.exception(traceback.format_exc())
    else:
        st.warning("Please enter a question.")

# ---- Feedback controls
if st.session_state.get("last_answer"):
    st.subheader("Provide feedback")
    col1, col2 = st.columns(2)

    if col1.button("👍 Correct"):
        log_feedback(
            query=st.session_state.last_query,
            answer=st.session_state.last_answer,
            feedback="up",
            feedback_dir=cfg.feedback_dir,
            session_id=st.session_state.session_id
        )
        st.success("Feedback recorded: 👍")
        try:
            log_mlflow_feedback(
                st.session_state.last_query,
                st.session_state.last_answer,
                "up",
                cfg,
                st.session_state.session_id
            )
        except Exception as e:
            st.warning(f"MLflow feedback not recorded: {e}")

    if col2.button("👎 Incorrect"):
        log_feedback(
            query=st.session_state.last_query,
            answer=st.session_state.last_answer,
            feedback="down",
            feedback_dir=cfg.feedback_dir,
            session_id=st.session_state.session_id
        )
        st.success("Feedback recorded: 👎")
        try:
            log_mlflow_feedback(
                st.session_state.last_query,
                st.session_state.last_answer,
                "down",
                cfg,
                st.session_state.session_id
            )
        except Exception as e:
            st.warning(f"MLflow feedback not recorded: {e}")

# ---- Add text docs
st.subheader("Add new text to knowledge base")
new_text = st.text_area("Enter text to store:")

if st.button("Add to Vector Store"):
    if new_text.strip():
        try:
            vectors = embedder.embed_texts([new_text])
            doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            text_store.add_texts(
                ids=[doc_id],
                texts=[new_text],
                embeddings=vectors,
                metadatas=[{"source": "ui"}]
            )

            if hasattr(text_store, "persist"):
                text_store.persist()

            save_dir = Path(cfg.storage_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = save_dir / f"{doc_id}.txt"
            filename.write_text(new_text, encoding="utf-8")
            st.success(f"Stored and saved as {doc_id}")
        except Exception as e:
            st.error(f"Failed to add text: {e}")
            st.exception(traceback.format_exc())
    else:
        st.warning("Please enter some text before adding.")