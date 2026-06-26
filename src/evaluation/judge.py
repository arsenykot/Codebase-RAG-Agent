import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def judge_response(query, response, reference):
    prompt = "Оцени ответ RAG-ассистента по шкале 1-5 (accuracy, completeness, usefulness).\n"
    prompt = prompt + "Верни JSON.\n\n"
    prompt = prompt + "Вопрос: " + query + "\n"
    prompt = prompt + "Эталон: " + reference + "\n"
    prompt = prompt + "Ответ: " + response

    api_key = os.getenv("OPENROUTER_API_KEY")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    result = client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    text = result.choices[0].message.content
    if text is None:
        text = ""
    return text
