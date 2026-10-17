"""
evaluation/ragas_eval.py

RAGAS-based evaluation of the RAG pipeline.
Computes: Faithfulness, Answer Relevancy, Context Precision, Context Recall.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class EvaluationSample:
    """A single evaluation data point."""
    question: str
    answer: str
    contexts: list[str]        # retrieved chunk texts
    ground_truth: str = ""     # optional reference answer


@dataclass
class EvaluationResult:
    """Aggregated RAGAS evaluation metrics."""
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    answer_correctness: float = 0.0

    def to_dict(self) -> dict:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "answer_correctness": round(self.answer_correctness, 4),
        }


class RAGASEvaluator:
    """
    Wraps the RAGAS library to evaluate RAG pipeline quality.

    Usage:
        evaluator = RAGASEvaluator()
        samples = [
            EvaluationSample(
                question="What does login() do?",
                answer="It authenticates users...",
                contexts=["def login(): ..."],
                ground_truth="login() validates credentials and returns a JWT",
            )
        ]
        result = evaluator.evaluate(samples)
        print(result.to_dict())
    """

    def __init__(self, llm=None, embeddings=None) -> None:
        self.llm = llm
        self.embeddings = embeddings
        self._ragas_available = self._check_ragas()

    @staticmethod
    def _check_ragas() -> bool:
        try:
            import ragas  # noqa
            return True
        except ImportError:
            logger.warning("RAGAS not installed. Install with: pip install ragas")
            return False

    def evaluate(self, samples: list[EvaluationSample]) -> EvaluationResult:
        """
        Run RAGAS evaluation on the provided samples.

        Falls back to a simple heuristic scorer if RAGAS is unavailable.
        """
        if not samples:
            raise ValueError("No evaluation samples provided.")

        if self._ragas_available:
            return self._ragas_evaluate(samples)
        else:
            logger.warning("Using heuristic evaluation (RAGAS not available).")
            return self._heuristic_evaluate(samples)

    def _ragas_evaluate(self, samples: list[EvaluationSample]) -> EvaluationResult:
        """Run actual RAGAS evaluation."""
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        data = {
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "contexts": [[c] for s in samples for c in [s.contexts]],
            "ground_truth": [s.ground_truth for s in samples],
        }
        dataset = Dataset.from_dict(data)

        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        if any(s.ground_truth for s in samples):
            metrics.append(answer_correctness)

        kwargs: dict[str, Any] = {}
        if self.llm:
            kwargs["llm"] = self.llm
        if self.embeddings:
            kwargs["embeddings"] = self.embeddings

        scores = evaluate(dataset, metrics=metrics, **kwargs)
        scores_dict = scores.to_pandas().mean().to_dict()

        return EvaluationResult(
            faithfulness=scores_dict.get("faithfulness", 0.0),
            answer_relevancy=scores_dict.get("answer_relevancy", 0.0),
            context_precision=scores_dict.get("context_precision", 0.0),
            context_recall=scores_dict.get("context_recall", 0.0),
            answer_correctness=scores_dict.get("answer_correctness", 0.0),
        )

    @staticmethod
    def _heuristic_evaluate(samples: list[EvaluationSample]) -> EvaluationResult:
        """Simple heuristic scoring when RAGAS is unavailable."""
        faithfulness_scores = []
        relevancy_scores = []

        for s in samples:
            # Faithfulness: fraction of answer tokens found in contexts
            ctx_text = " ".join(s.contexts).lower()
            ans_tokens = set(s.answer.lower().split())
            ctx_tokens = set(ctx_text.split())
            faith = len(ans_tokens & ctx_tokens) / max(len(ans_tokens), 1)
            faithfulness_scores.append(min(faith, 1.0))

            # Answer relevancy: overlap between question and answer
            q_tokens = set(s.question.lower().split())
            rel = len(q_tokens & ans_tokens) / max(len(q_tokens), 1)
            relevancy_scores.append(min(rel * 2, 1.0))  # scale up

        return EvaluationResult(
            faithfulness=sum(faithfulness_scores) / len(faithfulness_scores),
            answer_relevancy=sum(relevancy_scores) / len(relevancy_scores),
            context_precision=0.0,
            context_recall=0.0,
        )

    def save_results(self, result: EvaluationResult, output_path: Path) -> None:
        """Save evaluation results to a JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Evaluation results saved to {output_path}")
