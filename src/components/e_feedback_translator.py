import json
import re
import yaml
from src.components.llm_client import LLMClient


class FeedbackTranslator:
    def __init__(
        self,
        config_path="src/config/config.yaml",
        prompt_path=None,
    ):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        # Infer prompt_path from config_path if not provided
        if prompt_path is None:
            csv_path = self.cfg.get("data", {}).get("csv_path", "diabetes")
            dataset_name = (
                csv_path.split("/")[-1].replace("_cleaned.csv", "").replace(".csv", "")
            )
            prompt_path = f"src/prompts/prompts_{dataset_name}.yaml"

        with open(prompt_path, "r") as f:
            prompts = yaml.safe_load(f)
            self.prompt_template = prompts["system_prompts"]["feedback_translator"][
                "prompt"
            ]

        self.model_name = self.cfg["verifier"].get("llm_judge_model", "llama3")
        self.ollama = LLMClient.from_model(self.model_name)

        self.dimensions = ["technicality", "verbosity", "depth", "perspective"]

    def translate(self, feedback_text: str):
        prompt = self.prompt_template.replace("{{feedback}}", feedback_text.strip())
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }

        try:
            response = self.ollama.post(payload)
            response.raise_for_status()
            res_text = response.json().get("response", "{}")
        except Exception:
            res_text = "{}"

        deltas = self._safe_parse_deltas(res_text)
        return deltas

    def _safe_parse_deltas(self, text):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        raw = match.group() if match else "{}"
        try:
            data = json.loads(raw)
        except Exception:
            data = {}

        deltas = {}
        for dim in self.dimensions:
            val = data.get(dim, 0.0)
            try:
                num = float(val)
            except (TypeError, ValueError):
                num = 0.0
            deltas[dim] = max(-1.0, min(1.0, num))

        return deltas
