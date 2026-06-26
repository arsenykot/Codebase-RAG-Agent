import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

JUDGE_PROMPT = """Оцени ответ RAG-ассистента по анализу кода по трём критериям (1-5):
- accuracy: насколько ответ соответствует вопросу и фактам
- completeness: насколько полный ответ
- usefulness: насколько ответ полезен разработчику

Верни только JSON вида:
{{"accuracy": N, "completeness": N, "usefulness": N, "comment": "краткий комментарий"}}

Вопрос: {query}
Эталонный ответ: {reference}
Ответ ассистента: {response}
"""


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def judge_response(query: str, response: str, reference: str) -> str:
    client = _get_client()
    prompt = JUDGE_PROMPT.format(
        query=query,
        reference=reference,
        response=response,
    )
    result = client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return result.choices[0].message.content or ""
