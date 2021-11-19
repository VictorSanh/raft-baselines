from abc import ABC, abstractmethod
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import torch
import requests
import os

class Embedder(ABC):
    @abstractmethod
    def __call__(self, texts: List[str]) -> List[List[float]]:
        ...


class SentenceTransformersEmbedder(Embedder):
    def __init__(
        self, model_name="sentence-transformers/all-MiniLM-L6-v2", max_seq_length=512
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.similarity_model = SentenceTransformer(model_name, device=self.device)
        except requests.exceptions.ProxyError:
            model_path = os.path.join(os.getenv("SENTENCE_TRANSFORMERS_HOME"), model_name.replace("/", "_"))
            if os.path.exists(os.path.join(model_path, "modules.json")):
                dummy_class = SentenceTransformer()
                modules = dummy_class._load_sbert_model(model_path)
            else:
                raise ValueError(f"{os.path.join(model_path, 'modules.json')} does not exist")
            self.similarity_model = SentenceTransformer(modules=modules, device=self.device)
        self.similarity_model.max_seq_length = max_seq_length
        self._cache = {}

    def __call__(self, texts: Tuple[str]) -> List[List[float]]:
        if hash(texts) in self._cache:
            return self._cache[hash(texts)]

        embeds = self.similarity_model.encode(
            texts, convert_to_tensor=True, device=self.device
        )

        self._cache[hash(texts)] = embeds
        return embeds
