import os
from dataclasses import dataclass
from typing import Optional, List

import requests
from dotenv import load_dotenv

LOCAL_OLLAMA_URL = "http://localhost:11434/api/generate"
LOCAL_OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
CLOUD_OLLAMA_URL = "https://ollama.com/api/generate"


def is_cloud_model(model_name: str) -> bool:
    return "-cloud" in model_name or ":cloud" in model_name


@dataclass
class OllamaClient:
    model_name: str
    api_url: str
    api_key: Optional[str]

    @classmethod
    def from_model(cls, model_name: str, warn_if_missing_key: bool = True):
        load_dotenv()
        if is_cloud_model(model_name):
            api_url = CLOUD_OLLAMA_URL
            api_key = os.getenv("OLLAMA_API_KEY")
            if warn_if_missing_key and not api_key:
                print(
                    "[Warning] OLLAMA_API_KEY not set. Cloud models require authentication."
                )
                print(
                    "[Info] Set OLLAMA_API_KEY environment variable or run 'ollama signin'"
                )
        else:
            api_url = LOCAL_OLLAMA_URL
            api_key = None
        return cls(model_name=model_name, api_url=api_url, api_key=api_key)

    def _get_headers(self) -> dict:
        """Build request headers with optional auth."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def post(self, payload: dict, timeout: Optional[int] = None):
        response = requests.post(
            self.api_url, json=payload, headers=self._get_headers(), timeout=timeout
        )
        response.raise_for_status()
        return response

    def embed(
        self, text: str, model: str = "embeddinggemma", timeout: Optional[int] = 60
    ) -> List[float]:
        """
        Generate embeddings using Ollama's embedding endpoint.

        Args:
            text: Text to embed
            model: Ollama embedding model name (default: embeddinggemma)
            timeout: Request timeout in seconds

        Returns:
            List of floats representing the embedding vector
        """
        payload = {"model": model, "input": text}
        response = requests.post(
            LOCAL_OLLAMA_EMBED_URL,
            json=payload,
            headers=self._get_headers(),
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json()
        # Ollama returns {"embeddings": [[...]]} for single input
        return result["embeddings"][0]

    def embed_batch(
        self,
        texts: List[str],
        model: str = "embeddinggemma",
        timeout: Optional[int] = 120,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            model: Ollama embedding model name
            timeout: Request timeout in seconds

        Returns:
            List of embedding vectors
        """
        payload = {"model": model, "input": texts}
        response = requests.post(
            LOCAL_OLLAMA_EMBED_URL,
            json=payload,
            headers=self._get_headers(),
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json()
        return result["embeddings"]
