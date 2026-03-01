"""
Thin LLM client supporting Anthropic (default) and OpenAI backends.

Backend is selected via LLM_BACKEND env var ("anthropic" or "openai").
Model is selected via MODEL env var.
API keys from ANTHROPIC_API_KEY or OPENAI_API_KEY.
"""

import os


class LLMClient:
    def __init__(self):
        self.backend = os.environ.get("LLM_BACKEND", "anthropic").lower()
        self.model = os.environ.get("MODEL", self._default_model())
        self._client = self._init_client()

    def _default_model(self) -> str:
        if os.environ.get("LLM_BACKEND", "anthropic").lower() == "openai":
            return "gpt-4o-mini"
        return "claude-haiku-4-5-20251001"

    def _init_client(self):
        if self.backend == "anthropic":
            import anthropic
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            return anthropic.Anthropic(api_key=key)
        elif self.backend == "openai":
            import openai
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError("OPENAI_API_KEY is not set")
            return openai.OpenAI(api_key=key)
        raise ValueError(f"Unknown LLM_BACKEND: {self.backend!r}. Use 'anthropic' or 'openai'.")

    def completion(self, messages: list[dict]) -> str:
        """Send messages to the LLM and return the response text."""
        if self.backend == "anthropic":
            # Anthropic requires system message as a separate param
            system_content = None
            filtered = []
            for m in messages:
                if m["role"] == "system":
                    system_content = m["content"]
                else:
                    filtered.append(m)
            kwargs: dict = {
                "model": self.model,
                "max_tokens": 4096,
                "messages": filtered,
            }
            if system_content:
                kwargs["system"] = system_content
            resp = self._client.messages.create(**kwargs)
            return resp.content[0].text
        else:  # openai
            resp = self._client.chat.completions.create(
                model=self.model,
                max_tokens=4096,
                messages=messages,
            )
            return resp.choices[0].message.content
