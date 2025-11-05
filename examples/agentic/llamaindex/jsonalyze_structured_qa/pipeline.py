from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from llama_index.core.base.response.schema import Response
from llama_index.core.indices.struct_store.sql_retriever import DefaultSQLParser
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_JSONALYZE_PROMPT
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.settings import Settings
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from llama_index.core.workflow.events import Event
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv


def _load_env() -> None:
    candidates = []
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidates.append(parent / ".env")
    candidates.append(Path("/cvlization_repo/.env"))
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            load_dotenv(candidate, override=False)


_load_env()

try:
    import sqlite_utils
except ImportError:
    sqlite_utils = None

DATA_PATH = Path(__file__).resolve().parent / "data" / "people.json"
HF_CACHE = Path(os.getenv("HF_HOME", "/workspace/.cache/huggingface"))
DEFAULT_TABLE_NAME = "items"
DEFAULT_PROVIDER = os.getenv("LLAMA_JSONALYZE_PROVIDER", "mock").lower()
DEFAULT_OPENAI_MODEL = os.getenv("LLAMA_JSONALYZE_OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_HF_MODEL = os.getenv(
    "LLAMA_JSONALYZE_HF_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)
DEFAULT_EMBED_MODEL = os.getenv(
    "LLAMA_JSONALYZE_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)



DEFAULT_RESPONSE_SYNTHESIS_PROMPT_TMPL = (
    "Given a query, synthesize a response based on SQL query results"
    " to satisfy the query. Only include details that are relevant to"
    " the query. If you don't know the answer, then say that.\n"
    "SQL Query: {sql_query}\n"
    "Table Schema: {table_schema}\n"
    "SQL Response: {sql_response}\n"
    "Query: {query_str}\n"
    "Response: "
)

DEFAULT_RESPONSE_SYNTHESIS_PROMPT = PromptTemplate(
    DEFAULT_RESPONSE_SYNTHESIS_PROMPT_TMPL,
    prompt_type=PromptType.SQL_RESPONSE_SYNTHESIS,
)


class JSONAnalyzeQueryEngineWorkflow(Workflow):
    @step
    async def jsonalyzer(self, ctx: Context, ev: StartEvent) -> Event:
        if sqlite_utils is None:
            raise ImportError(
                "sqlite-utils is required. Rebuild the Docker image or install sqlite-utils."
            )

        await ctx.store.set("query", ev.get("query"))
        await ctx.store.set("llm", ev.get("llm"))

        query = ev.get("query")
        table_name = ev.get("table_name")
        list_of_dict = ev.get("list_of_dict")
        prompt = DEFAULT_JSONALYZE_PROMPT

        db = sqlite_utils.Database(memory=True)
        try:
            db[table_name].insert_all(list_of_dict)
        except sqlite_utils.utils.sqlite3.IntegrityError as exc:
            raise ValueError("Invalid list_of_dict") from exc

        table_schema = db[table_name].columns_dict
        response_str = await ev.llm.apredict(
            prompt=prompt,
            table_name=table_name,
            table_schema=table_schema,
            question=query,
        )

        sql_parser = DefaultSQLParser()
        sql_query = sql_parser.parse_response_to_sql(response_str, ev.query)

        try:
            raw_results = list(db.query(sql_query))
            results = [dict(row) for row in raw_results]
        except sqlite_utils.utils.sqlite3.OperationalError as exc:
            raise ValueError(f"Invalid query generated: {sql_query}") from exc

        return Event(
            sql_query=sql_query, table_schema=table_schema, results=results
        )

    @step
    async def synthesize(self, ctx: Context, ev: Event) -> StopEvent:
        llm = await ctx.store.get("llm")
        query = await ctx.store.get("query")
        response_str = llm.predict(
            DEFAULT_RESPONSE_SYNTHESIS_PROMPT,
            sql_query=ev.sql_query,
            table_schema=ev.table_schema,
            sql_response=ev.results,
            query_str=query,
        )
        response_metadata = {
            "sql_query": ev.sql_query,
            "table_schema": str(ev.table_schema),
            "results": ev.results,
        }
        response = Response(response=response_str, metadata=response_metadata)
        return StopEvent(result=response)


def _ensure_hf_cache() -> None:
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_CACHE))


def _embedding() -> HuggingFaceEmbedding:
    _ensure_hf_cache()
    embed = HuggingFaceEmbedding(model_name=DEFAULT_EMBED_MODEL)
    Settings.embed_model = embed
    return embed


def _load_people() -> List[Dict[str, Any]]:
    return json.loads(DATA_PATH.read_text())


def _configure_llm(provider: str):
    provider = provider.lower()
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for provider 'openai'.")
        llm = OpenAI(model=DEFAULT_OPENAI_MODEL)
        return llm
    if provider in {"hf", "huggingface"}:
        _ensure_hf_cache()
        llm = HuggingFaceLLM(
            model_name=DEFAULT_HF_MODEL,
            tokenizer_name=DEFAULT_HF_MODEL,
            generate_kwargs={"max_new_tokens": 128},
        )
        return llm
    raise ValueError(f"Unsupported provider '{provider}'.")


async def _arun_workflow(llm, question: str, data: List[Dict[str, Any]], table_name: str):
    workflow = JSONAnalyzeQueryEngineWorkflow()
    handler = workflow.run(
        query=question, list_of_dict=data, llm=llm, table_name=table_name
    )
    return await handler


def _run_workflow(llm, question: str, data: List[Dict[str, Any]], table_name: str):
    return asyncio.run(_arun_workflow(llm, question, data, table_name))


MOCK_SQL_PATTERNS = [
    (
        lambda q: "maximum age" in q,
        "SELECT MAX(age) AS max_age FROM items",
        lambda rows, total: f"The maximum age in the dataset is {rows[0]['max_age']}.",
    ),
    (
        lambda q: "occupation" in q and any(word in q for word in ["science", "engineering"]),
        """
        SELECT COUNT(*) AS count
        FROM items
        WHERE LOWER(occupation) LIKE '%science%'
           OR LOWER(occupation) LIKE '%engineer%'
        """,
        lambda rows, total: f"There are {rows[0]['count']} individuals working in science or engineering."
        ,
    ),
    (
        lambda q: "starting" in q and "+1 234" in q,
        """
        SELECT COUNT(*) AS count
        FROM items
        WHERE phone LIKE '+1 234%'
        """,
        lambda rows, total: f"{rows[0]['count']} individuals have phone numbers starting with +1 234."
        ,
    ),
    (
        lambda q: "percentage" in q and ("california" in q or " ca" in q),
        """
        SELECT
            ROUND(100.0 * SUM(CASE WHEN state = 'CA' THEN 1 ELSE 0 END) / COUNT(*), 2)
            AS percentage
        FROM items
        """,
        lambda rows, total: f"Approximately {rows[0]['percentage']}% of individuals reside in California."
        ,
    ),
    (
        lambda q: "major" in q and "psychology" in q,
        """
        SELECT COUNT(*) AS count FROM items WHERE LOWER(major) = 'psychology'
        """,
        lambda rows, total: f"{rows[0]['count']} individuals have a major in Psychology."
        ,
    ),
]


def _run_mock(question: str, data: List[Dict[str, Any]], table_name: str):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    columns = data[0].keys()
    placeholders = ",".join("?" for _ in columns)
    conn.execute(
        f"CREATE TABLE {table_name} ({', '.join(columns)})"
    )
    conn.executemany(
        f"INSERT INTO {table_name} VALUES ({placeholders})",
        [[row[col] for col in columns] for row in data],
    )
    conn.commit()

    question_lower = question.lower()
    for matcher, sql, formatter in MOCK_SQL_PATTERNS:
        if matcher(question_lower):
            rows = [dict(r) for r in conn.execute(sql).fetchall()]
            answer = formatter(rows, len(data))
            return {
                "mode": "mock",
                "answer": answer,
                "sql_query": sql.strip(),
                "results": rows,
            }

    return {
        "mode": "mock",
        "answer": "No heuristic available for this question in mock mode.",
        "sql_query": None,
        "results": [],
    }


def run_query(question: str, provider: str | None = None) -> Dict[str, Any]:
    provider = (provider or DEFAULT_PROVIDER).lower()
    data = _load_people()
    table_name = DEFAULT_TABLE_NAME

    if provider == "mock":
        return _run_mock(question, data, table_name)

    llm = _configure_llm(provider)
    result = _run_workflow(llm, question, data, table_name)
    if isinstance(result, Response):
        metadata = result.metadata or {}
        sql_query = metadata.get("sql_query")
        rows = metadata.get("results") or []
        return {
            "mode": provider,
            "answer": str(result).strip(),
            "sql_query": sql_query,
            "results": rows,
        }

    # For safety if workflow returns StopEvent
    if isinstance(result, StopEvent):
        payload = result.result  # type: ignore[attr-defined]
        if isinstance(payload, Response):
            metadata = payload.metadata or {}
            return {
                "mode": provider,
                "answer": str(payload).strip(),
                "sql_query": metadata.get("sql_query"),
                "results": metadata.get("results") or [],
            }
    raise RuntimeError("Unexpected workflow result")
