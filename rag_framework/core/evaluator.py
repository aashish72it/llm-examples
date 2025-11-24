import mlflow
from rag_framework.utils.errors import EvaluationError
from rag_framework.core.llm import LLMClient

# Keep adapter + HF embeddings internal to Evaluator for Ragas compatibility
from rag_framework.core.llm_adapter import LLMAdapter  # if you prefer, move this into core and import here
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset

class Evaluator:
    def __init__(self, tracking_uri: str, experiment_name: str):
        try:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)

            self.llm_client = LLMClient()
            self.llm_adapter = LLMAdapter(self.llm_client)

            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception as e:
            raise EvaluationError(f"Setup failed: {e}")

    def evaluate(
        self,
        query: str,
        context_docs: list[str],
        answer: str,
        ground_truth: str | None = None
    ) -> dict:
        try:
            from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
            from ragas import evaluate as ragas_evaluate

            data = Dataset.from_dict({
                "question": [query],
                "contexts": [context_docs],
                "answer": [answer],
                "ground_truth": [ground_truth or ""]
            })

            result = ragas_evaluate(
                data,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                llm=self.llm_adapter,
                embeddings=self.embeddings
            )
            df = result.to_pandas()
            metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
            scores = {metric: float(df[metric][0]) for metric in metric_cols if metric in df.columns}

            with mlflow.start_run(run_name=f"eval_{query[:30]}"):
                mlflow.log_params({
                    "has_ground_truth": bool(ground_truth),
                    "num_contexts": len(context_docs),
                })
                mlflow.log_metrics(scores)

            prompt = f"""
You are an evaluator for a RAG pipeline.
Query: {query}
Contexts: {context_docs}
Answer: {answer}

Provide a short qualitative assessment of the answer quality.
            """
            qualitative = self.llm_client.ask_llm([{"role": "user", "content": prompt}])

            return {"scores": scores, "qualitative": qualitative}
        except Exception as e:
            raise EvaluationError(f"Evaluation failed: {e}")
