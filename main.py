import argparse
from pathlib import Path

from src.chunking import chunk_python_files
from src.evaluation.run import run_evaluation
from src.generation import answer_question
from src.indexing import build_index, get_indexed_repo_path, load_index
from src.ingestion import load_python_files


def resolve_repo_path(explicit_path: str | Path | None = None) -> Path:
    if explicit_path is not None:
        path = Path(explicit_path).resolve()
        if not path.is_dir():
            raise ValueError(f"Repository path does not exist: {path}")
        return path

    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    repo_drop = data_dir / "repo"
    if repo_drop.is_dir() and any(
        item.name != ".gitkeep" for item in repo_drop.iterdir()
    ):
        return repo_drop.resolve()

    ignored = {"faiss_index", "repo"}
    candidates = sorted(
        (p for p in data_dir.iterdir() if p.is_dir() and p.name not in ignored),
        key=lambda p: p.name,
    )

    if not candidates:
        raise ValueError(
            "Положите папку репозитория в data/\n"
            "  • data/repo/  (рекомендуется)\n"
            "  • или data/имя-проекта/\n"
            "Затем запустите: python main.py index"
        )

    if len(candidates) == 1:
        return candidates[0].resolve()

    names = ", ".join(p.name for p in candidates)
    raise ValueError(
        f"В data/ несколько папок-репозиториев: {names}. "
        "Оставьте одну или укажите путь явно: python main.py index <path>"
    )


def cmd_index(repo_path: str | None = None) -> None:
    path = resolve_repo_path(repo_path)
    print(f"Indexing repository: {path}")
    files = load_python_files(path)
    print(f"Loaded {len(files)} Python files")

    chunks = chunk_python_files(files)
    print(f"Created {len(chunks)} chunks")

    build_index(chunks, path)
    print(f"Index saved to data/faiss_index")


def cmd_ask() -> None:
    repo = get_indexed_repo_path()
    if repo:
        print(f"Indexed repository: {repo}")
    else:
        load_index()

    print("Codebase RAG Agent. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Bye.")
            break

        try:
            answer = answer_question(question)
            print(f"\nAnswer:\n{answer}\n")
        except Exception as exc:
            print(f"Error: {exc}\n")


def cmd_evaluate(k: int, no_judge: bool) -> None:
    results = run_evaluation(k=k, use_judge=not no_judge)
    print("\n=== Evaluation summary ===")
    print(f"Mean Recall@{k}: {results['mean_recall_at_k']:.2%}")
    if results["judge_results"]:
        print(f"LLM judge runs: {len(results['judge_results'])}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Codebase RAG Agent — analyze Python repositories"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Build FAISS index from a repo")
    index_parser.add_argument(
        "repo_path",
        nargs="?",
        default=None,
        help="Path to repo (default: auto-detect folder in data/)",
    )

    subparsers.add_parser("ask", help="Interactive Q&A in terminal")

    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation metrics")
    eval_parser.add_argument(
        "--k", type=int, default=5, help="k for Recall@k (default: 5)"
    )
    eval_parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM-as-a-Judge (retrieval metrics only)",
    )

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args.repo_path)
    elif args.command == "ask":
        cmd_ask()
    elif args.command == "evaluate":
        cmd_evaluate(args.k, args.no_judge)


if __name__ == "__main__":
    main()
