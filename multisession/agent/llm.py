"""One chat interface for every model: the OpenAI-compatible Chat Completions API.

GPT models use api.openai.com; local models run behind Ollama's compatible
endpoint.  Same messages, tool schemas, temperature, and token limit for all,
so differences come from the model, not from the plumbing.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


PROVIDERS = {
    "openai": lambda: OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
    "ollama": lambda: OpenAI(base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"), api_key="ollama"),
}


class ChatModel:
    def __init__(self, provider: str, model: str, temperature: float = 0.0, max_tokens: int = 2048, seed: int = 0):
        _load_dotenv()
        self.provider, self.model = provider, model
        self.client = PROVIDERS[provider]()
        self.params = {"temperature": temperature, "max_tokens": max_tokens, "seed": seed}

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    def chat(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> Tuple[Dict, Dict]:
        kwargs = dict(model=self.model, messages=messages, **self.params)
        if tools:
            kwargs["tools"] = tools
        tick = time.perf_counter()
        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        usage = response.usage
        out = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            out["tool_calls"] = [{"id": c.id, "type": "function",
                                  "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                 for c in message.tool_calls]
        return out, {"prompt_tokens": getattr(usage, "prompt_tokens", 0),
                     "completion_tokens": getattr(usage, "completion_tokens", 0),
                     "seconds": time.perf_counter() - tick}
