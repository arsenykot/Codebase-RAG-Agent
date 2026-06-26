import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "ollama"
DEFAULT_MODEL = "llama3.2"


def get_client():
    base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL)
    api_key = os.getenv("OLLAMA_API_KEY", DEFAULT_API_KEY)
    return OpenAI(base_url=base_url, api_key=api_key)


def get_model():
    return os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)


def chat_completion(messages, temperature=0.3):
    response = get_client().chat.completions.create(
        model=get_model(),
        messages=messages,
        temperature=temperature,
    )
    content = response.choices[0].message.content
    if content is None:
        return ""
    return content
