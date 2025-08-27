import json
from typing import List, Optional

import requests

try:  # Support running as package (e.g., app.config) or module
    from config import settings  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for packaged apps
    from app.config import settings  # type: ignore


class OllamaClient:
    def __init__(self, host: Optional[str] = None):
        self.host = (host or settings.OLLAMA_HOST).rstrip("/")

    def generate(
        self,
        prompt: str,
        model: str = "llama3:8b",
        temperature: float = 0.2,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
    ) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            payload["stop"] = stop
        url = f"{self.host}/api/generate"
        response_text = ""
        with requests.post(url, json=payload, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                response_text += data.get("response", "")
        return response_text
