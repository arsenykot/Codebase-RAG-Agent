from src.llm import chat_completion


def judge_response(query, response, reference):
    prompt = "Оцени ответ RAG-ассистента по шкале 1-5 (accuracy, completeness, usefulness).\n"
    prompt = prompt + "Верни JSON.\n\n"
    prompt = prompt + "Вопрос: " + query + "\n"
    prompt = prompt + "Эталон: " + reference + "\n"
    prompt = prompt + "Ответ: " + response

    return chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
