import json
from typing import List, Optional

from urllib import request as urlrequest

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
        request_data = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(url, data=request_data, headers={"Content-Type": "application/json"})
        response_text = ""
        with urlrequest.urlopen(req) as resp:
            for raw in resp:
                if not raw:
                    continue
                data = json.loads(raw.decode("utf-8"))
                response_text += data.get("response", "")
        return response_text
