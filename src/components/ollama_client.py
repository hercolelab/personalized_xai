import os
from dataclasses import dataclass
from typing import Optional, List

import requests
from dotenv import load_dotenv

LOCAL_OLLAMA_URL = "http://localhost:11434/api/generate"
LOCAL_OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
CLOUD_OLLAMA_URL = "https://ollama.com/api/generate"
GROQ_OPENAI_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_CHAT_COMPLETIONS_URL = f"{GROQ_OPENAI_BASE_URL}/chat/completions"


def is_cloud_model(model_name: str) -> bool:
    return "-cloud" in model_name or ":cloud" in model_name


def is_groq_model(model_name: str) -> bool:
    return model_name.startswith("groq:") or model_name.startswith("groq/")


def strip_groq_prefix(model_name: str) -> str:
    if model_name.startswith("groq:"):
        return model_name[len("groq:") :]
    if model_name.startswith("groq/"):
        return model_name[len("groq/") :]
    return model_name


class _CompatResponse:
    def __init__(self, content: str, raw: dict, status_code: int = 200):
        self._content = content
        self._raw = raw
        self.status_code = status_code

    def json(self) -> dict:
        return {"response": self._content, "raw": self._raw}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


@dataclass
class OllamaClient:
    model_name: str
    api_url: str
    api_key: Optional[str]
    provider: str = "ollama"

    @classmethod
    def from_model(cls, model_name: str, warn_if_missing_key: bool = True):
        load_dotenv()
        if is_groq_model(model_name):
            api_url = GROQ_CHAT_COMPLETIONS_URL
            api_key = os.getenv("GROQ_API_KEY")
            if warn_if_missing_key and not api_key:
                print("[Warning] GROQ_API_KEY not set. Groq requires authentication.")
                print(
                    "[Info] Set GROQ_API_KEY environment variable for Groq access."
                )
            return cls(
                model_name=strip_groq_prefix(model_name),
                api_url=api_url,
                api_key=api_key,
                provider="groq",
            )

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

    def _build_groq_payload(self, payload: dict) -> dict:
        prompt = payload.get("prompt", "")
        temperature = None
        options = payload.get("options")
        if isinstance(options, dict):
            temperature = options.get("temperature")
        if temperature is None:
            temperature = payload.get("temperature")

        groq_payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            groq_payload["temperature"] = temperature
        return groq_payload

    def post(self, payload: dict, timeout: Optional[int] = None):
        if self.provider == "groq":
            groq_payload = self._build_groq_payload(payload)
            response = requests.post(
                self.api_url,
                json=groq_payload,
                headers=self._get_headers(),
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            message = (
                data.get("choices", [{}])[0].get("message", {}) or {}
            ).get("content", "")
            return _CompatResponse(message, data, status_code=response.status_code)

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
