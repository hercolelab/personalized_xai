import os
from dataclasses import dataclass
from typing import Optional

import requests
from dotenv import load_dotenv

LOCAL_OLLAMA_URL = "http://localhost:11434/api/generate"
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

    def post(self, payload: dict, timeout: Optional[int] = None):
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(
            self.api_url, json=payload, headers=headers, timeout=timeout
        )
        response.raise_for_status()
        return response
