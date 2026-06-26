from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def _data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_index(chunks: list[Document], repo_path: str | Path) -> FAISS:
    embeddings = _get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    index_dir = _data_dir() / "faiss_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    (_data_dir() / "repo_path.txt").write_text(
        str(Path(repo_path).resolve()), encoding="utf-8"
    )
    return vectorstore


def load_index() -> FAISS:
    index_dir = _data_dir() / "faiss_index"
    if not index_dir.exists():
        raise FileNotFoundError(
            "FAISS index not found. Run: python main.py index <path_to_repo>"
        )

    embeddings = _get_embeddings()
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_indexed_repo_path() -> str | None:
    meta_file = _data_dir() / "repo_path.txt"
    if meta_file.exists():
        return meta_file.read_text(encoding="utf-8").strip()
    return None
