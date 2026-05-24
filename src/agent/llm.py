from __future__ import annotations

import logging
from typing import Iterator, Optional

import ollama

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Thin wrapper around the Ollama Python SDK.
    Supports streaming and non-streaming chat completions.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
        num_ctx: int = 8192,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.num_ctx = num_ctx
        self._client = ollama.Client(host=base_url)

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    def check_connection(self) -> bool:
        """Return True if Ollama is reachable."""
        try:
            self._client.list()
            return True
        except Exception as exc:
            logger.error("Cannot reach Ollama at %s: %s", self.base_url, exc)
            return False

    def check_model(self) -> bool:
        """Return True if the configured model is pulled and available."""
        try:
            response = self._client.list()
            available = [m.model or "" for m in response.models]
            # Accept partial match so "qwen2.5:14b" matches "qwen2.5:14b-instruct-q4_K_M"
            return any(self.model in m for m in available)
        except Exception as exc:
            logger.error("Could not list Ollama models: %s", exc)
            return False

    def list_models(self) -> list[str]:
        """Return names of all locally available models."""
        try:
            response = self._client.list()
            return [m.model or "" for m in response.models]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        stream: bool = True,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        options = {
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
        }

        try:
            if stream:
                return self._stream(messages, options)
            return self._complete(messages, options)
        except Exception as exc:
            raise RuntimeError(f"LLM generation failed: {exc}") from exc

    def _stream(self, messages: list, options: dict) -> str:
        full: list[str] = []
        for chunk in self._client.chat(
            model=self.model,
            messages=messages,
            options=options,
            stream=True,
        ):
            token = chunk.message.content or ""
            print(token, end="", flush=True)
            full.append(token)
        print()  # trailing newline after stream ends
        return "".join(full)

    def _complete(self, messages: list, options: dict) -> str:
        response = self._client.chat(
            model=self.model,
            messages=messages,
            options=options,
            stream=False,
        )
        return response.message.content or ""
