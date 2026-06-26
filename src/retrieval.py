from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

_cross_encoder: CrossEncoder | None = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def retrieve(vectorstore: FAISS, question: str, k: int = 20) -> list[Document]:
    return vectorstore.similarity_search(question, k=k)


def rerank(
    question: str,
    candidates: list[Document],
    top_k: int = 5,
) -> list[Document]:
    if not candidates:
        return []

    model = _get_cross_encoder()
    pairs = [(question, doc.page_content) for doc in candidates]
    scores = model.predict(pairs)

    ranked = sorted(
        zip(candidates, scores, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    return [doc for doc, _score in ranked[:top_k]]
