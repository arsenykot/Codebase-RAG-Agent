from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_index(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = FAISS.from_documents(chunks, embeddings)

    base = Path(__file__).resolve().parent.parent / "data"
    index_path = base / "faiss_index"
    index_path.mkdir(parents=True, exist_ok=True)

    vectorstore.save_local(str(index_path))

    return vectorstore


def load_index():
    base = Path(__file__).resolve().parent.parent / "data"
    index_path = base / "faiss_index"

    if not index_path.exists():
        print("Сначала запустите: python main.py index")
        raise FileNotFoundError("Индекс не найден")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore
