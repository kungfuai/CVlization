from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd

from smolagents import CodeAgent, PythonInterpreterTool
from smolagents.tools import Tool
from smolagents.models import LiteLLMModel

DEFAULT_DATA_PATH = "data/marketing_kpi.csv"


@dataclass
class ProviderConfig:
    provider: str
    model_id: Optional[str]
    temperature: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Smolagents data analyst over a marketing dataset.")
    parser.add_argument("--question", required=True, type=str, help="User question to analyze against the dataset.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to the CSV dataset (defaults to bundled marketing_kpi.csv)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON payload instead of plain text.")
    parser.add_argument(
        "--llm-provider",
        type=str,
        help="Override the LLM provider (mock, openai, groq, ollama). Defaults to mock if unspecified.",
    )
    parser.add_argument("--llm-model", type=str, help="Override the provider-specific model identifier.")
    parser.add_argument("--temperature", type=float, help="Override sampling temperature.")
    return parser.parse_args()


def detect_provider(args: argparse.Namespace) -> ProviderConfig:
    provider = (
        (args.llm_provider or os.getenv("ANALYST_LLM_PROVIDER") or os.getenv("SMOL_LLM_PROVIDER"))
        or os.getenv("LLM_PROVIDER")
        or "mock"
    ).lower()
    model_id = args.llm_model or os.getenv("ANALYST_LLM_MODEL") or os.getenv("LLM_MODEL")
    temp_env = (
        args.temperature
        if args.temperature is not None
        else os.getenv("ANALYST_LLM_TEMPERATURE")
        or os.getenv("LLM_TEMPERATURE")
    )
    try:
        temperature = float(temp_env) if temp_env is not None else 0.2
    except ValueError:
        temperature = 0.2
    return ProviderConfig(provider=provider, model_id=model_id, temperature=temperature)


def load_dataset(path: str) -> pd.DataFrame:
    resolved = os.path.join(os.path.dirname(__file__), path) if not os.path.isabs(path) else path
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Dataset not found at {resolved}")
    df = pd.read_csv(resolved)
    if df.empty:
        raise ValueError("Dataset is empty; please provide a CSV with at least one row.")
    return df


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def run_rule_based_answer(question: str, df: pd.DataFrame) -> str:
    """Fallback deterministic summary so the example works without an LLM."""
    question_lower = question.lower()
    segment_totals = df.groupby("segment")["revenue"].sum().sort_values(ascending=False)
    region_totals = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    top_customer = df.sort_values("revenue", ascending=False).iloc[0]
    total_revenue = df["revenue"].sum()

    summary_lines = [
        "Here is a quick analysis of the marketing KPI dataset:",
        f"• Total revenue across all customers: {format_currency(total_revenue)}.",
        f"• Top performing segment: {segment_totals.index[0]} with {format_currency(segment_totals.iloc[0])} in revenue.",
        f"• Strongest region: {region_totals.index[0]} contributing {format_currency(region_totals.iloc[0])}.",
        f"• Highest value customer: {top_customer['customer_id']} ({top_customer['segment']}) generating {format_currency(top_customer['revenue'])}.",
    ]

    if "segment" in question_lower:
        summary_lines.append("Segment breakdown (revenue): " + ", ".join(
            f"{seg}: {format_currency(val)}" for seg, val in segment_totals.items()
        ))
    if "region" in question_lower or "geo" in question_lower:
        summary_lines.append("Regional breakdown (revenue): " + ", ".join(
            f"{reg}: {format_currency(val)}" for reg, val in region_totals.items()
        ))
    return "\n".join(summary_lines)


class DuckDBSQLTool(Tool):
    name = "duckdb_sql"
    description = (
        "Executes a SQL query against the marketing_kpi table registered in DuckDB. "
        "Use this tool to aggregate, filter, or join data when answering business questions."
    )
    inputs = {"query": {"type": "string", "description": "SQL query to execute against marketing_kpi."}}
    output_type = "string"

    def __init__(self, connection: duckdb.DuckDBPyConnection):
        super().__init__()
        self.connection = connection

    def forward(self, query: str) -> str:
        try:
            df = self.connection.execute(query).fetchdf()
        except Exception as exc:  # pragma: no cover - delegated to agent behaviour
            return f"Query failed: {exc}"
        if df.empty:
            return "Query returned no rows."
        return df.to_markdown(index=False)


def build_model(config: ProviderConfig) -> LiteLLMModel:
    provider = config.provider
    model_id = config.model_id
    temperature = config.temperature

    if provider in {"mock", "fake"}:
        raise ValueError("Mock provider should not instantiate an LLM model.")

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY must be set for provider 'openai'.")
        model_id = model_id or "openai/gpt-4o-mini"
        return LiteLLMModel(model_id=model_id, api_key=api_key, temperature=temperature)

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY must be set for provider 'groq'.")
        model_id = model_id or "groq/llama3-8b-8192"
        return LiteLLMModel(model_id=model_id, api_key=api_key, temperature=temperature)

    if provider == "ollama":
        api_base = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        model_id = model_id or "ollama/llama3.1"
        return LiteLLMModel(model_id=model_id, api_base=api_base, temperature=temperature)

    raise ValueError(
        "Unsupported provider '{provider}'. Supported values: mock, openai, groq, ollama."
    )


def run_agentic_answer(question: str, df: pd.DataFrame, config: ProviderConfig) -> str:
    connection = duckdb.connect(database=":memory:")
    connection.register("marketing_kpi", df)

    tools: List[Tool] = [
        DuckDBSQLTool(connection),
        PythonInterpreterTool(authorized_imports=["pandas", "duckdb"]),
    ]

    model = build_model(config)

    instructions = (
        "You are an experienced data analyst. The marketing_kpi table is already registered in DuckDB. "
        "Prefer the duckdb_sql tool for aggregations and filtering. When sharing results, cite specific figures and keep the conclusion concise."
    )

    agent = CodeAgent(
        tools=tools,
        model=model,
        additional_authorized_imports=["pandas", "duckdb"],
        instructions=instructions,
        max_steps=8,
    )

    try:
        result = agent.run(question)
    finally:
        connection.close()
    return str(result)


def main() -> None:
    args = parse_args()
    config = detect_provider(args)
    df = load_dataset(args.data_path)

    if config.provider in {"mock", "fake"}:
        answer = run_rule_based_answer(args.question, df)
        meta: Dict[str, Any] = {
            "mode": "rule_based",
            "message": "Set --llm-provider (or ANALYST_LLM_PROVIDER env var) to leverage smolagents with a real model.",
        }
    else:
        try:
            answer = run_agentic_answer(args.question, df, config)
            meta = {"mode": "smolagents", "provider": config.provider, "model_id": config.model_id}
        except Exception as exc:  # pragma: no cover - ensures graceful fallback
            answer = (
                "Falling back to deterministic summary because the agent run failed. "
                f"Reason: {exc}\n\n"
            ) + run_rule_based_answer(args.question, df)
            meta = {"mode": "fallback", "error": str(exc)}

    if args.json:
        payload: Dict[str, Any] = {"question": args.question, "answer": answer, "metadata": meta}
        print(json.dumps(payload, indent=2))
    else:
        print(f"Q: {args.question}\nA: {answer}")
        if meta.get("mode") != "smolagents":
            print(f"\n[info] {meta['message'] if 'message' in meta else 'Agent run did not use an external LLM.'}")


if __name__ == "__main__":
    main()
