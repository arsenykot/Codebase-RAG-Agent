# Codebase RAG Agent

Агент, который анализирует исходный код Python-репозитория и отвечает на вопросы о его структуре, логике и взаимосвязях между модулями.

## Тема проекта

RAG-система для анализа кода: загрузка `.py` файлов → структурный AST-чанкинг → FAISS → reranking → ответ LLM.

## Быстрый старт

```bash
pip install -r requirements.txt
cp .env.example .env
# Добавьте OPENROUTER_API_KEY в .env

# 1. Положите Python-репозиторий в data/repo/ (или data/имя-проекта/)
# 2. Проиндексируйте и задавайте вопросы:
python main.py index
python main.py ask
python main.py evaluate --no-judge   # только Recall@k (без API)
python main.py evaluate              # Recall@k + LLM-as-a-Judge
```

При первом запуске скачиваются модели `sentence-transformers` и `cross-encoder` (~100–200 MB).

## Команды

| Команда | Описание |
|---------|----------|
| `python main.py index` | Проиндексировать репозиторий из `data/` |
| `python main.py index <path>` | Проиндексировать репозиторий по явному пути |
| `python main.py ask` | Интерактивные вопросы в терминале |
| `python main.py evaluate` | Метрики Recall@k и LLM-as-a-Judge |

## Соответствие критериям рубрики

| Критерий | Баллы | Реализация |
|----------|-------|------------|
| Тема проекта | 5 | README, агент для анализа Python-кода |
| Сбор документов | 5 | `src/ingestion.py` — обход `.py` файлов |
| Ingestion + чанкинг | 15 | `ingestion.py` + `chunking.py` |
| Структурный чанкинг | 10 | `src/chunking.py` — AST по функциям/классам/методам |
| Retriever | 15 | `src/retrieval.py` — `similarity_search` |
| Vector DB | 10 | `src/indexing.py` — FAISS + `all-MiniLM-L6-v2` |
| Reranking | 10 | `src/retrieval.py` — CrossEncoder top-5 |
| LLM chain | 15 | `src/generation.py` — retrieve → rerank → prompt → OpenRouter |
| Evaluation | 15 | `src/evaluation/` — Recall@k + LLM-as-a-Judge |

## Архитектура

```
repo/*.py → ingestion → chunking → FAISS index
question → retrieve (k=20) → rerank (top-5) → LLM → answer
```

## Демо-репозиторий

Положите Python-проект в [`data/repo/`](data/repo/) и выполните `python main.py index`.

Для тестов самого агента можно скопировать папку `src/` в `data/repo/`. Тестовые вопросы — в `src/evaluation/test_questions.json`.

## Примеры вопросов

- Как загружаются Python-файлы из репозитория?
- Как реализован структурный чанкинг Python-кода?
- Как работает reranking после retrieval?
- Какая модель эмбеддингов используется?

## Структура проекта

```
src/
├── ingestion.py
├── chunking.py
├── indexing.py
├── retrieval.py
├── generation.py
└── evaluation/metrics.py, judge.py, test_questions.json
```

## Источники из курса

- `lessons/student_rag4.ipynb`, `student_rag6.ipynb` — FAISS, chunking, RAG chain
- `lessons/teacher_rag7.ipynb` — Recall@k, LLM-as-a-Judge
- `lessons/teacher_rag8.ipynb` — структурный чанкинг (концепция), CrossEncoder reranking
