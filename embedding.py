import os
from typing import List

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class NomicOnnxEmbedder:
    """ONNXRuntime wrapper for nomic-ai/nomic-embed-text local model."""

    def __init__(self, model_dir: str, max_len: int = None, onnx_filename: str = "model.onnx"):
        self.model_dir = model_dir.rstrip("/")
        tok_path = os.path.join(self.model_dir, "tokenizer.json")
        onnx_path = os.path.join(self.model_dir, "onnx", onnx_filename)

        if not os.path.exists(tok_path):
            raise RuntimeError(f"Tokenizer not found: {tok_path}")
        if not os.path.exists(onnx_path):
            raise RuntimeError(f"ONNX model not found: {onnx_path}")

        self.tokenizer = Tokenizer.from_file(tok_path)
        self.max_len = max_len or int(os.getenv("EMBED_MAX_LEN", "2048"))

        # CPU by default; switch to onnxruntime-gpu + CUDA providers if desired
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)

        # Model I/O signatures
        self.input_names = [i.name for i in self.session.get_inputs()]
        outs = [o.name for o in self.session.get_outputs()]
        if not outs:
            raise RuntimeError("Could not resolve ONNX output name.")
        self.output_name = outs[0]  # often 'last_hidden_state'

    def _prepare_arrays(self, texts: List[str]):
        max_len = self.max_len
        input_ids, attention_mask = [], []
        for t in texts:
            enc = self.tokenizer.encode(t or "")
            ids = enc.ids[:max_len]
            att = [1] * len(ids)
            if len(ids) < max_len:
                pad = max_len - len(ids)
                ids += [0] * pad
                att += [0] * pad
            input_ids.append(ids)
            attention_mask.append(att)

        # Optional inputs: many BERT-style models require these
        token_type_ids = [[0] * max_len for _ in texts]          # all zeros
        position_ids   = [list(range(max_len)) for _ in texts]   # sometimes expected

        arrays = {
            "input_ids":      np.asarray(input_ids, dtype=np.int64),
            "attention_mask": np.asarray(attention_mask, dtype=np.int64),
            "token_type_ids": np.asarray(token_type_ids, dtype=np.int64),
            "position_ids":   np.asarray(position_ids, dtype=np.int64),
        }
        return arrays

    def _masked_mean_pool(self, token_embs: np.ndarray, attn: np.ndarray) -> np.ndarray:
        attn = attn[:, :token_embs.shape[1]].astype(np.float32)
        attn_exp = np.expand_dims(attn, axis=-1)                 # [B, T, 1]
        summed = (token_embs * attn_exp).sum(axis=1)             # [B, H]
        counts = np.clip(attn_exp.sum(axis=1), 1e-6, None)       # [B, 1]
        return summed / counts

    def encode(self, texts, normalize_embeddings: bool = True):
        if not isinstance(texts, list):
            texts = [texts]

        arrays = self._prepare_arrays(texts)

        # Feed ONLY what the model actually requires
        feed = {name: arrays[name] for name in self.input_names if name in arrays}
        missing = [name for name in self.input_names if name not in feed]
        if missing:
            raise RuntimeError(f"Missing inputs not covered by adapter: {missing}")

        out = self.session.run([self.output_name], feed)[0]
        vecs = np.asarray(out, dtype=np.float32)

        # Pool if the model returned token embeddings
        if vecs.ndim == 3:                     # [B, T, H]
            vecs = self._masked_mean_pool(vecs, arrays["attention_mask"])
        elif vecs.ndim == 4:
            vecs = vecs.squeeze()
            if vecs.ndim == 3:
                vecs = self._masked_mean_pool(vecs, arrays["attention_mask"])
        elif vecs.ndim == 1:                   # [H] -> [1, H]
            vecs = vecs[None, :]

        # L2 normalize
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
            vecs = vecs / norms

        return vecs.tolist()
