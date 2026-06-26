from langchain_core.documents import Document


def recall_at_k(
    retrieved: list[Document],
    expected_file: str | None = None,
    expected_symbol: str | None = None,
    k: int = 5,
) -> float:
    top_k = retrieved[:k]
    for doc in top_k:
        file_path = doc.metadata.get("file_path", "")
        symbol_name = doc.metadata.get("symbol_name", "")
        file_match = expected_file is None or expected_file in file_path
        symbol_match = expected_symbol is None or expected_symbol == symbol_name
        if file_match and symbol_match:
            return 1.0
    return 0.0


def mean_recall_at_k(scores: list[float]) -> float:
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
