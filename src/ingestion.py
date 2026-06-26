from pathlib import Path


def _should_skip(path: Path) -> bool:
    skip_dirs = {
        ".git",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        "node_modules",
        "lessons",
        "data",
    }
    return any(part in skip_dirs for part in path.parts)


def load_python_files(repo_path: str | Path) -> list[dict]:
    root = Path(repo_path).resolve()
    if not root.is_dir():
        raise ValueError(f"Repository path does not exist: {root}")

    files: list[dict] = []
    for file_path in sorted(root.rglob("*.py")):
        if _should_skip(file_path.relative_to(root)):
            continue
        try:
            source = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            source = file_path.read_text(encoding="latin-1")

        files.append(
            {
                "file_path": str(file_path.relative_to(root)).replace("\\", "/"),
                "absolute_path": str(file_path),
                "source": source,
            }
        )

    if not files:
        raise ValueError(f"No Python files found in repository: {root}")

    return files
