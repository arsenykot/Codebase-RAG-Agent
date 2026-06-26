import json
from pathlib import Path

from src.evaluation.judge import judge_response
from src.evaluation.metrics import mean_recall_at_k, recall_at_k
from src.generation import answer_question
from src.indexing import load_index
from src.retrieval import retrieve


def run_evaluation(k: int = 5, use_judge: bool = True) -> dict:
    questions_path = Path(__file__).parent / "test_questions.json"
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    vectorstore = load_index()

    recall_scores: list[float] = []
    judge_results: list[dict] = []

    print(f"\nEvaluating {len(questions)} questions (Recall@{k})...\n")

    for item in questions:
        query = item["question"]
        expected_file = item.get("expected_file")
        expected_symbol = item.get("expected_symbol")
        reference = item.get("reference_answer", "")

        retrieved = retrieve(vectorstore, query, k=k)
        score = recall_at_k(
            retrieved,
            expected_file=expected_file,
            expected_symbol=expected_symbol,
            k=k,
        )
        recall_scores.append(score)
        status = "HIT" if score == 1.0 else "MISS"
        print(f"[{status}] {query}")

        if use_judge and reference:
            try:
                answer = answer_question(query)
                judge_text = judge_response(query, answer, reference)
                judge_results.append(
                    {
                        "question": query,
                        "answer": answer,
                        "judge": judge_text,
                    }
                )
                print(f"  Judge: {judge_text[:120]}...")
            except Exception as exc:
                judge_results.append(
                    {
                        "question": query,
                        "error": str(exc),
                    }
                )
                print(f"  Judge skipped: {exc}")

    mean_recall = mean_recall_at_k(recall_scores)
    print(f"\nMean Recall@{k}: {mean_recall:.2%}")

    return {
        "mean_recall_at_k": mean_recall,
        "recall_scores": recall_scores,
        "judge_results": judge_results,
    }
