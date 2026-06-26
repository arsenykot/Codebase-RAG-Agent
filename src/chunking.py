import ast

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def fallback_chunk_file(file_path: str, source: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    doc = Document(
        page_content=source,
        metadata={
            "file_path": file_path,
            "symbol_name": file_path,
            "symbol_type": "file",
            "start_line": 1,
        },
    )
    chunks = splitter.split_documents([doc])
    for chunk in chunks:
        chunk.metadata.setdefault("file_path", file_path)
        chunk.metadata.setdefault("symbol_type", "file")
    return chunks


def _get_source_segment(source: str, node: ast.AST) -> str:
    try:
        return ast.get_source_segment(source, node) or ""
    except (TypeError, ValueError):
        lines = source.splitlines()
        start = max(getattr(node, "lineno", 1) - 1, 0)
        end = getattr(node, "end_lineno", start + 1)
        return "\n".join(lines[start:end])


def _make_document(
    source: str,
    node: ast.AST,
    file_path: str,
    symbol_name: str,
    symbol_type: str,
    parent_class: str | None = None,
) -> Document:
    code = _get_source_segment(source, node)
    docstring = ast.get_docstring(node) or ""
    content_parts = [f"# {file_path}"]
    if parent_class:
        content_parts.append(f"# class: {parent_class}")
    content_parts.append(f"# {symbol_type}: {symbol_name}")
    if docstring:
        content_parts.append(f'"""{docstring}"""')
    content_parts.append(code)

    metadata = {
        "file_path": file_path,
        "symbol_name": symbol_name,
        "symbol_type": symbol_type,
        "start_line": getattr(node, "lineno", 0),
    }
    if parent_class:
        metadata["parent_class"] = parent_class

    return Document(page_content="\n".join(content_parts), metadata=metadata)


def _extract_from_class(
    source: str,
    class_node: ast.ClassDef,
    file_path: str,
) -> list[Document]:
    documents = [
        _make_document(
            source,
            class_node,
            file_path,
            class_node.name,
            "class",
        )
    ]

    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            documents.append(
                _make_document(
                    source,
                    item,
                    file_path,
                    item.name,
                    "method",
                    parent_class=class_node.name,
                )
            )

    return documents


def chunk_python_file(file_path: str, source: str) -> list[Document]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return fallback_chunk_file(file_path, source)

    documents: list[Document] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            documents.append(
                _make_document(
                    source,
                    node,
                    file_path,
                    node.name,
                    "function",
                )
            )
        elif isinstance(node, ast.ClassDef):
            documents.extend(_extract_from_class(source, node, file_path))

    if not documents:
        return fallback_chunk_file(file_path, source)

    return documents


def chunk_python_files(files: list[dict]) -> list[Document]:
    chunks: list[Document] = []
    for file_info in files:
        chunks.extend(
            chunk_python_file(file_info["file_path"], file_info["source"])
        )
    return chunks
