from pathlib import Path


def load_python_files(repo_path):
    root = Path(repo_path)
    if not root.exists():
        print("Папка не найдена:", repo_path)
        return []

    files = []

    for file_path in root.rglob("*.py"):
        rel_path = file_path.relative_to(root)
        skip = False

        parts = str(rel_path).replace("\\", "/").split("/")
        for part in parts:
            if part in [".git", "venv", ".venv", "__pycache__", "lessons", "data", "node_modules"]:
                skip = True
                break

        if skip:
            continue

        f = open(file_path, "r", encoding="utf-8")
        source = f.read()
        f.close()

        files.append({
            "file_path": str(rel_path).replace("\\", "/"),
            "source": source,
        })

    if len(files) == 0:
        print("Python файлов не найдено!")

    return files
