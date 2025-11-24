import os
import json
import uuid
import mlflow
from datetime import datetime
import streamlit as st

from multimodal.utils.errors import FeedbackError


def log_feedback(query, answer, feedback, feedback_dir, session_id=None):
    """Write feedback to a per-session JSON file in feedback_dir."""
    session_id = session_id or str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"session_{session_id}_{timestamp}.json"
    filepath = os.path.join(feedback_dir, filename)

    record = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "query": query,
        "answer": answer,
        "feedback": feedback
    }

    try:
        os.makedirs(feedback_dir, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        st.write(f"Feedback recorded in {filename}")
    except Exception as e:
        raise FeedbackError(f"Failed to log feedback locally: {e}")


def log_mlflow_feedback(query, answer, feedback, cfg, session_id):
    """Log feedback into MLflow for experiment tracking."""
    try:
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        mlflow.set_experiment(cfg.mlflow_experiment)
        with mlflow.start_run(run_name=f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param("query", query)
            mlflow.log_param("answer", answer)
            mlflow.log_param("feedback", feedback)
            mlflow.log_param("session_id", session_id)
            mlflow.log_metric("human_feedback", 1 if feedback == "up" else 0)
    except Exception as e:
        raise FeedbackError(f"Failed to log feedback to MLflow: {e}")