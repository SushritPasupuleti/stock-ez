from __future__ import annotations

import json
import logging
from typing import Iterator, Optional

import httpx
import ollama

# Local LLMs can take several minutes to generate a full analysis over a large
# context window.  Use a short connect timeout (fail fast if Ollama isn't
# running) and a generous read timeout that covers even the slowest 7B model
# on CPU-only hardware.
_TIMEOUT = httpx.Timeout(connect=5.0, read=600.0, write=30.0, pool=5.0)

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
        self._client = ollama.Client(host=base_url, timeout=_TIMEOUT)

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

    def token_stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """Yield tokens one at a time — designed for use with st.write_stream()."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        options = {"temperature": self.temperature, "num_ctx": self.num_ctx}
        for chunk in self._client.chat(
            model=self.model,
            messages=messages,
            options=options,
            stream=True,
        ):
            yield chunk.message.content or ""

    def pull_model_stream(
        self,
    ) -> Iterator[tuple[str, float]]:
        """Pull (download) the model, yielding (status_message, fraction) tuples.

        fraction is in [0.0, 1.0] — best-effort based on byte progress.
        Designed for driving a st.progress() bar in the Streamlit UI.
        """
        for chunk in self._client.pull(model=self.model, stream=True):
            status: str = chunk.status or ""
            total = getattr(chunk, "total", None)
            completed = getattr(chunk, "completed", None)
            if total and completed:
                fraction = min(float(completed) / float(total), 1.0)
            elif status == "success":
                fraction = 1.0
            elif status in ("verifying sha256 digest", "writing manifest"):
                fraction = 0.99
            else:
                fraction = 0.0
            yield status, fraction


# ─────────────────────────────────────────────────────────────────────────────
# vLLM — OpenAI-compatible backend
# ─────────────────────────────────────────────────────────────────────────────

# Maps Ollama model names → HuggingFace model IDs and recommended settings.
# AWQ INT4 variants are preferred for the NVIDIA 4060 Ti 16 GB.
# HF IDs may need updating as new model releases appear on HuggingFace.
VLLM_MODEL_CONFIG: dict[str, dict] = {
    "qwen3:14b":                    {"hf_id": "Qwen/Qwen3-14B-AWQ",                              "quant": "awq",  "vram_gb": 9},
    "qwen3.5:9b":                   {"hf_id": "Qwen/Qwen3.5-9B-Instruct-AWQ",                   "quant": "awq",  "vram_gb": 6},
    "qwen3:8b":                     {"hf_id": "Qwen/Qwen3-8B-AWQ",                               "quant": "awq",  "vram_gb": 6},
    "mistral-small3.2:24b":         {"hf_id": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",   "quant": "gptq", "vram_gb": 14},
    "deepseek-r1:14b-distill-qwen": {"hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",        "quant": "awq",  "vram_gb": 9},
    "deepseek-r1:14b":              {"hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",        "quant": "awq",  "vram_gb": 9},
    "gpt-oss:20b":                  {"hf_id": "openai/gpt-oss-20b",                              "quant": "awq",  "vram_gb": 11},
    "phi4:14b":                     {"hf_id": "microsoft/phi-4-awq",                             "quant": "awq",  "vram_gb": 8},
    "gemma3:27b":                   {"hf_id": "google/gemma-3-27b-it",                           "quant": "awq",  "vram_gb": 15},
    "gemma3:12b":                   {"hf_id": "google/gemma-3-12b-it",                           "quant": "awq",  "vram_gb": 8},
}


def build_vllm_serve_cmd(
    model_name: str,
    vllm_url: str = "http://localhost:8000",
    gpu_util: float = 0.88,
    cpu_offload_gb: int = 16,
    max_model_len: int = 32768,
) -> str:
    """Return the optimal `vllm serve` command for NVIDIA 4060 Ti 16 GB + 32 GB RAM.

    - FP16 arithmetic (fast on Ampere/Ada, avoids BF16 compatibility issues)
    - AWQ or GPTQ quantization to fit the model inside 16 GB VRAM
    - 88 % GPU utilisation (leaves ~1.9 GB headroom for KV-cache spikes)
    - 16 GB CPU offload via --cpu-offload-gb (uses system RAM for overflow layers)
    - --enforce-eager disables CUDA-graph capture, saving ~1-2 GB on first load
    """
    cfg = VLLM_MODEL_CONFIG.get(model_name, {})
    hf_id = cfg.get("hf_id", model_name)
    quant = cfg.get("quant", "awq")
    try:
        port = vllm_url.rstrip("/").rsplit(":", 1)[-1]
        int(port)  # validate it's actually a port number
    except (ValueError, IndexError):
        port = "8000"
    lines = [
        f"vllm serve {hf_id}",
        f"  --dtype float16",
        f"  --quantization {quant}",
        f"  --gpu-memory-utilization {gpu_util}",
        f"  --cpu-offload-gb {cpu_offload_gb}",
        f"  --max-model-len {max_model_len}",
        f"  --enforce-eager",
        f"  --tensor-parallel-size 1",
        f"  --port {port}",
    ]
    _sep = " \\\n"
    return _sep.join(lines)


class VLLMClient:
    """Client for vLLM's OpenAI-compatible REST API.

    Connects to a running `vllm serve` instance and streams chat completions
    via Server-Sent Events (SSE).  Drop-in replacement for OllamaClient —
    both expose the same ``token_stream`` interface used by st.write_stream().
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000",
        temperature: float = 0.3,
        num_ctx: int = 8192,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.num_ctx = num_ctx  # sent as max_tokens to the API

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    def check_connection(self) -> bool:
        """Return True if the vLLM server is reachable."""
        try:
            resp = httpx.get(
                f"{self.base_url}/v1/models",
                timeout=httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0),
            )
            return resp.status_code == 200
        except Exception as exc:
            logger.error("Cannot reach vLLM at %s: %s", self.base_url, exc)
            return False

    def get_loaded_models(self) -> list[str]:
        """Return model IDs currently served by this vLLM instance."""
        try:
            resp = httpx.get(
                f"{self.base_url}/v1/models",
                timeout=httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0),
            )
            data = resp.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception:
            return []

    def check_model(self) -> bool:
        """Return True if self.model is loaded in the running vLLM instance."""
        loaded = self.get_loaded_models()
        # Accept partial match: 'Qwen/Qwen3-14B-AWQ' matches 'qwen3-14b-awq'
        model_lower = self.model.lower().replace("/", "-")
        for m in loaded:
            m_lower = m.lower().replace("/", "-")
            if model_lower in m_lower or m_lower in model_lower:
                return True
        return False

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def token_stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """Yield tokens one at a time via SSE — designed for st.write_stream()."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.num_ctx,
            "stream": True,
        }
        with httpx.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=_TIMEOUT,
            headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                chunk_str = line[len("data: "):]
                if chunk_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(chunk_str)
                    delta = chunk["choices"][0]["delta"]
                    token = delta.get("content") or ""
                    if token:
                        yield token
                except (KeyError, json.JSONDecodeError):
                    continue
