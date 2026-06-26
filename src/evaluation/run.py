import json
from pathlib import Path

from src.evaluation.judge import judge_response
from src.evaluation.metrics import mean_recall_at_k, recall_at_k
from src.generation import answer_question
from src.indexing import load_index
from src.retrieval import retrieve


def run_evaluation(k=5, use_judge=True):
    path = Path(__file__).parent / "test_questions.json"
    f = open(path, "r", encoding="utf-8")
    questions = json.load(f)
    f.close()

    vectorstore = load_index()

    recall_scores = []
    judge_results = []

    print()
    print("Тестируем", len(questions), "вопросов")
    print()

    for item in questions:
        q = item["question"]
        expected_file = item.get("expected_file")
        expected_symbol = item.get("expected_symbol")
        reference = item.get("reference_answer", "")

        found = retrieve(vectorstore, q, k=k)
        score = recall_at_k(found, expected_file, expected_symbol, k)

        recall_scores.append(score)

        if score == 1.0:
            print("[OK]", q)
        else:
            print("[--]", q)

        if use_judge == True and reference != "":
            try:
                ans = answer_question(q)
                judge_text = judge_response(q, ans, reference)
                judge_results.append({"question": q, "answer": ans, "judge": judge_text})
                print("  judge:", judge_text[:100])
            except Exception as e:
                print("  judge error:", e)

    avg = mean_recall_at_k(recall_scores)
    print()
    print("Средний Recall@" + str(k) + ":", round(avg * 100, 1), "%")

    return {
        "mean_recall_at_k": avg,
        "recall_scores": recall_scores,
        "judge_results": judge_results,
    }
