import argparse
from pathlib import Path

from src.chunking import chunk_python_files
from src.evaluation.run import run_evaluation
from src.generation import answer_question
from src.indexing import build_index
from src.ingestion import load_python_files


def resolve_repo_path(explicit_path=None):
    if explicit_path is not None:
        path = Path(explicit_path).resolve()
        if not path.is_dir():
            print("Ошибка: такой папки нет")
            raise ValueError(f"Repository path does not exist: {path}")
        return path

    data_dir = Path(__file__).resolve().parent / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True)

    repo_folder = data_dir / "repo"
    has_files = False
    if repo_folder.exists():
        for item in repo_folder.iterdir():
            if item.name != ".gitkeep":
                has_files = True
                break

    if has_files:
        return repo_folder.resolve()

    folders = []
    for item in data_dir.iterdir():
        if item.is_dir() and item.name != "faiss_index" and item.name != "repo":
            folders.append(item)

    if len(folders) == 0:
        print("Положите репозиторий в data/repo/")
        raise ValueError("Нет репозитория в data/")

    if len(folders) == 1:
        return folders[0].resolve()

    names = ""
    for f in folders:
        names = names + f.name + ", "
    raise ValueError(f"В data/ несколько папок: {names}. Оставьте одну.")


def cmd_index(repo_path=None):
    path = resolve_repo_path(repo_path)
    print("Индексируем:", path)

    files = load_python_files(path)
    print("Нашли файлов:", len(files))

    chunks = chunk_python_files(files)
    print("Получилось чанков:", len(chunks))

    build_index(chunks)


def cmd_ask():
    print()
    print("Задавайте вопросы. Для выхода: exit")
    print()

    while True:
        question = input("Вопрос: ").strip()

        if question == "":
            continue

        if question == "exit" or question == "quit" or question == "q":
            print("Пока!")
            break

        try:
            answer = answer_question(question)
            print()
            print("Ответ:")
            print(answer)
            print()
        except Exception as e:
            print("Ошибка:", e)
            print()


def cmd_evaluate(k, no_judge):
    use_judge = True
    if no_judge:
        use_judge = False

    run_evaluation(k=k, use_judge=use_judge)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    index_parser = subparsers.add_parser("index")
    index_parser.add_argument("repo_path", nargs="?", default=None)

    subparsers.add_parser("ask")

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--k", type=int, default=5)
    eval_parser.add_argument("--no-judge", action="store_true")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args.repo_path)
    elif args.command == "ask":
        cmd_ask()
    elif args.command == "evaluate":
        cmd_evaluate(args.k, args.no_judge)
    else:
        print("Команды: index, ask, evaluate")


if __name__ == "__main__":
    main()
