import os

from dotenv import load_dotenv
from openai import OpenAI

from src.indexing import load_index
from src.retrieval import rerank, retrieve

load_dotenv()


def answer_question(question):
    vectorstore = load_index()

    candidates = retrieve(vectorstore, question, k=20)

    docs = rerank(question, candidates, top_k=5)

    context = ""
    for doc in docs:
        fp = doc.metadata.get("file_path", "?")
        sym = doc.metadata.get("symbol_name", "?")
        stype = doc.metadata.get("symbol_type", "?")
        context = context + "[" + fp + " | " + stype + ": " + sym + "]\n"
        context = context + doc.page_content + "\n\n"

    prompt = (
        "Ты — помощник, анализирующий Python-репозиторий. Отвечай на основе приведённого ниже контекста: никаких домыслов или информации вне его.\n"
        "Всегда ссылайся на конкретные файлы, функции, классы или методы из контекста в своём ответе.\n"
        "Если необходимой информации в контексте нет, честно напиши: \"Информация не найдена\".\n\n"
        "Контекст:\n" + context + "\n"
        "Вопрос: " + question + "\n"
        "Ответ:"
    )

    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key is None or api_key == "":
        return "Ошибка: добавьте OPENROUTER_API_KEY в .env"

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    answer = response.choices[0].message.content
    if answer is None:
        answer = "Информация не найдена."

    return answer
