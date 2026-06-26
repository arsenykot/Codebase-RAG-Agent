import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from openai import OpenAI

from src.indexing import load_index
from src.retrieval import rerank, retrieve

load_dotenv()

PROMPT_TEMPLATE = """Ты анализируешь Python-репозиторий. Отвечай только по контексту.
Указывай файлы и символы (функции/классы). Если данных нет — скажи "Информация не найдена".

Контекст:
{context}

Вопрос: {question}
Ответ:"""


def _format_context(docs: list[Document]) -> str:
    parts: list[str] = []
    for doc in docs:
        file_path = doc.metadata.get("file_path", "unknown")
        symbol_name = doc.metadata.get("symbol_name", "")
        symbol_type = doc.metadata.get("symbol_type", "")
        header = f"[{file_path} | {symbol_type}: {symbol_name}]"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def answer_question(question: str) -> str:
    vectorstore = load_index()
    candidates = retrieve(vectorstore, question)
    docs = rerank(question, candidates)
    context = _format_context(docs)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    client = _get_client()
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content or "Информация не найдена."
