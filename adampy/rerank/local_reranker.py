from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import settings


class LocalCrossEncoderReranker:
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or settings.RERANKER_MODEL_PATH
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path, local_files_only=True)
        except Exception as e:  # pragma: no cover - will raise for missing model
            raise RuntimeError(f"Failed to load reranker model at {self.model_path}: {e}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def score(self, query: str, passages: List[str]) -> List[float]:
        scores: List[float] = []
        for i in range(0, len(passages), settings.RERANKER_BATCH_SIZE):
            batch = passages[i : i + settings.RERANKER_BATCH_SIZE]
            enc = self.tokenizer(
                text=[query] * len(batch),
                text_pair=batch,
                truncation=True,
                max_length=settings.RERANKER_MAX_LEN,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits.squeeze(-1)
            scores.extend(torch.sigmoid(logits).detach().cpu().tolist())
        return scores
