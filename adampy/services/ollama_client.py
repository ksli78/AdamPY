import logging
import json
from typing import List, Optional

from urllib import request as urlrequest

try:  # Support running as package (e.g., app.config) or module
    from config import settings  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for packaged apps
    from app.config import settings  # type: ignore

logger = logging.getLogger("rag")  # configure in main.py to output to journald or console

class OllamaClient:
    def __init__(self, host: Optional[str] = None):
        self.host = (host or settings.OLLAMA_HOST).rstrip("/")

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        # Backward-compatible params (explicit args override settings)
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,          # kept for compatibility
        stop: Optional[List[str]] = None,
        # New optional knobs (None = take from settings)
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        num_predict: Optional[int] = None,
        seed: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        mirostat: Optional[int] = None,
        mirostat_tau: Optional[float] = None,
        mirostat_eta: Optional[float] = None,
    ) -> str:
        
        mdl = model or settings.CHAT_MODEL
        opts = {
            "temperature": settings.OLLAMA_TEMPERATURE if temperature is None else temperature,
            "top_p": settings.OLLAMA_TOP_P if top_p is None else top_p,
            "top_k": settings.OLLAMA_TOP_K if top_k is None else top_k,
            "repeat_penalty": settings.OLLAMA_REPEAT_PENALTY if repeat_penalty is None else repeat_penalty,
            "num_predict": (
                # prefer explicit num_predict, then max_tokens, then settings
                (num_predict if num_predict is not None else max_tokens)
                if (num_predict is not None or max_tokens is not None)
                else settings.OLLAMA_NUM_PREDICT
            ),
            "presence_penalty": settings.OLLAMA_PRESENCE_PENALTY if presence_penalty is None else presence_penalty,
            "frequency_penalty": settings.OLLAMA_FREQUENCY_PENALTY if frequency_penalty is None else frequency_penalty,
            "mirostat": settings.OLLAMA_MIROSTAT if mirostat is None else mirostat,
            "mirostat_tau": settings.OLLAMA_MIROSTAT_TAU if mirostat_tau is None else mirostat_tau,
            "mirostat_eta": settings.OLLAMA_MIROSTAT_ETA if mirostat_eta is None else mirostat_eta,
            "stream":False
        }

        # Build payload
        payload = {
            "model": mdl,
            "prompt": prompt,
            "options": {k: v for k, v in opts.items() if v is not None},
        }

        # Stops: explicit arg wins; otherwise from settings (if provided)
        stops = stop if stop is not None else getattr(settings, "OLLAMA_STOP", None)
        if stops:
            payload["stop"] = stops

        # Keep-alive (optional; often set at model load time, but safe here)
        if getattr(settings, "OLLAMA_KEEP_ALIVE", None):
            payload["keep_alive"] = settings.OLLAMA_KEEP_ALIVE

        url = f"http://127.0.0.1:11434/api/generate"
        logger.info("Sending request to Ollama: url=%s payload=%s", url, json.dumps(payload, ensure_ascii=False))

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
# class OllamaClient:
#     def __init__(self, host: Optional[str] = None):
#         # If no host is provided, this falls back to a settings file.
#         # self.host = (host or settings.OLLAMA_HOST).rstrip("/")
#         # For demonstration, let's use a default value directly:
#         self.host = (host or "http://localhost:11434").rstrip("/")

#     def generate(
#         self,
#         prompt: str,
#         model: str = "llama3:8b",
#         temperature: float = 0.2,
#         max_tokens: int = 512,
#         stop: Optional[List[str]] = None,
#     ) -> str:
#         """
#         Generates a response from the Ollama API.
#         """
#         payload = {
#             "model": model,
#             "prompt": prompt,
#             "stream": False,  # Added for a single, complete response
#             "options": {
#                 "temperature": temperature,
#                 "num_predict": max_tokens,
#                 "stop": stop or [],
#             }
#         }
        
#         url = f"{self.host}/api/generate"
#         request_data = json.dumps(payload).encode("utf-8")
#         req = urlrequest.Request(
#             url,
#             data=request_data,
#             headers={"Content-Type": "application/json"}
#         )

#         with urlrequest.urlopen(req) as response:
#             response_body = response.read().decode("utf-8")
#             data = json.loads(response_body)
#             return data.get("response", "")