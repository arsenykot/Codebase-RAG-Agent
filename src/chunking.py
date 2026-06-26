import ast

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def fallback_chunk_file(file_path, source):
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    doc = Document(page_content=source, metadata={"file_path": file_path, "symbol_name": file_path, "symbol_type": "file"})
    chunks = splitter.split_documents([doc])
    return chunks


def chunk_python_file(file_path, source):
    try:
        tree = ast.parse(source)
    except:
        return fallback_chunk_file(file_path, source)

    documents = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            code = ast.get_source_segment(source, node)
            if code is None:
                code = ""

            text = "# " + file_path + "\n"
            text = text + "# function: " + node.name + "\n"
            text = text + code

            doc = Document(
                page_content=text,
                metadata={
                    "file_path": file_path,
                    "symbol_name": node.name,
                    "symbol_type": "function",
                },
            )
            documents.append(doc)

        elif isinstance(node, ast.ClassDef):
            code = ast.get_source_segment(source, node)
            if code is None:
                code = ""

            text = "# " + file_path + "\n"
            text = text + "# class: " + node.name + "\n"
            text = text + code

            doc = Document(
                page_content=text,
                metadata={
                    "file_path": file_path,
                    "symbol_name": node.name,
                    "symbol_type": "class",
                },
            )
            documents.append(doc)

            for item in node.body:
                if isinstance(item, ast.FunctionDef) or isinstance(item, ast.AsyncFunctionDef):
                    method_code = ast.get_source_segment(source, item)
                    if method_code is None:
                        method_code = ""

                    method_text = "# " + file_path + "\n"
                    method_text = method_text + "# class: " + node.name + "\n"
                    method_text = method_text + "# method: " + item.name + "\n"
                    method_text = method_text + method_code

                    method_doc = Document(
                        page_content=method_text,
                        metadata={
                            "file_path": file_path,
                            "symbol_name": item.name,
                            "symbol_type": "method",
                            "parent_class": node.name,
                        },
                    )
                    documents.append(method_doc)

    if len(documents) == 0:
        return fallback_chunk_file(file_path, source)

    return documents


def chunk_python_files(files):
    all_chunks = []
    for file_info in files:
        file_chunks = chunk_python_file(file_info["file_path"], file_info["source"])
        for ch in file_chunks:
            all_chunks.append(ch)
    return all_chunks
