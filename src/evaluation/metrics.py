def recall_at_k(retrieved, expected_file=None, expected_symbol=None, k=5):
    count = 0
    for doc in retrieved:
        if count >= k:
            break
        count = count + 1

        fp = doc.metadata.get("file_path", "")
        sym = doc.metadata.get("symbol_name", "")

        ok_file = True
        ok_sym = True

        if expected_file is not None:
            if expected_file not in fp:
                ok_file = False

        if expected_symbol is not None:
            if expected_symbol != sym:
                ok_sym = False

        if ok_file and ok_sym:
            return 1.0

    return 0.0


def mean_recall_at_k(scores):
    if len(scores) == 0:
        return 0.0
    total = 0.0
    for s in scores:
        total = total + s
    return total / len(scores)
