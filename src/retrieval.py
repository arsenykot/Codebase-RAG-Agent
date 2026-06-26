from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def retrieve(vectorstore, question, k=20):
    docs = vectorstore.similarity_search(question, k=k)
    return docs


def rerank(question, candidates, top_k=5):
    if len(candidates) == 0:
        return []

    model = CrossEncoder(RERANKER_MODEL)

    pairs = []
    for doc in candidates:
        pairs.append((question, doc.page_content))

    scores = model.predict(pairs)

    result = []
    for i in range(len(candidates)):
        result.append((candidates[i], scores[i]))

    n = len(result)
    for i in range(n):
        for j in range(0, n - i - 1):
            if result[j][1] < result[j + 1][1]:
                temp = result[j]
                result[j] = result[j + 1]
                result[j + 1] = temp

    top_docs = []
    for i in range(min(top_k, len(result))):
        top_docs.append(result[i][0])

    return top_docs
