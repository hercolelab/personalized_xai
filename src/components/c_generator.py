import json
import yaml
from src.components.ollama_client import OllamaClient


class NarrativeGenerator:
    def __init__(
        self, model_name="gpt-oss:20b-cloud", config_path="src/prompts/prompts.yaml"
    ):
        self.model_name = model_name
        self.ollama = OllamaClient.from_model(self.model_name)

        # Load the externalized prompts
        with open(config_path, "r") as f:
            full_config = yaml.safe_load(f)
            self.prompts = full_config["system_prompts"]["narrative_generator"]

    def _get_dynamic_instructions(self, style):
        """Helper to build style instructions based on numeric thresholds."""
        instr = []
        sf = self.prompts["style_guidelines"]["style_features"]

        # Technicality Logic
        if style["technicality"] < 0.4:
            instr.append(
                f"TECHNICALITY LOW: {sf['technicality']['low']['description']}"
            )
        elif style["technicality"] > 0.7:
            instr.append(
                f"TECHNICALITY HIGH: {sf['technicality']['high']['description']}"
            )
        else:
            instr.append(
                f"TECHNICALITY MEDIUM: {sf['technicality']['medium']['description']}"
            )

        # Verbosity Logic
        if style["verbosity"] < 0.4:
            instr.append(f"VERBOSITY LOW: {sf['verbosity']['low']['description']}")
        elif style["verbosity"] > 0.7:
            instr.append(f"VERBOSITY HIGH: {sf['verbosity']['high']['description']}")
        else:
            instr.append(
                f"VERBOSITY MEDIUM: {sf['verbosity']['medium']['description']}"
            )

        # Depth Logic
        if style["depth"] < 0.4:
            instr.append(f"DEPTH LOW: {sf['depth']['low']['description']}")
        elif style["depth"] > 0.7:
            instr.append(f"DEPTH HIGH: {sf['depth']['high']['description']}")
        else:
            instr.append(f"DEPTH MEDIUM: {sf['depth']['medium']['description']}")

        # Perspective Logic
        if style["perspective"] < 0.4:
            instr.append(
                f"PERSPECTIVE LOW: {sf['perspective']['low']['description']} Tone: {sf['perspective']['low']['tone']}"
            )
        elif style["perspective"] > 0.7:
            instr.append(
                f"PERSPECTIVE HIGH: {sf['perspective']['high']['description']} Tone: {sf['perspective']['high']['tone']}"
            )
        else:
            instr.append(
                f"PERSPECTIVE MEDIUM: {sf['perspective']['medium']['description']} Tone: {sf['perspective']['medium']['tone']}"
            )

        return "\n".join([f"- {i}" for i in instr])

    def generate(self, raw_data, style, hint=None):
        dynamic_style = self._get_dynamic_instructions(style)
        data_json = json.dumps(raw_data, indent=2)

        # Build the final prompt using YAML content
        prompt = self.prompts["prompt_template"].format(
            data_json=data_json, dynamic_style=dynamic_style
        )

        if hint:
            prompt += f"\n\n# CRITICAL CORRECTION REQUIRED:\n{hint}"

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},  # Lower temp for better data fidelity
        }

        try:
            response = self.ollama.post(payload)
            response.raise_for_status()
            res_text = response.json()["response"]

            if "NARRATIVE:" in res_text:
                return res_text.split("NARRATIVE:")[1].strip()
            return res_text
        except Exception as e:
            return f"Narrative Generation Error: {str(e)}"


# Example
if __name__ == "__main__":
    sample_data = {
        "current": {"Glucose": 148.0, "BMI": 33.6, "Age": 50.0},
        "target": {"Glucose": 110.0, "BMI": 28.5},
        "shap_impacts": {
            "Glucose": 0.12,
            "BMI": 0.08,
            "Age": 0.05,
            "BloodPressure": 0.03,
        },
        "probabilities": {"current": 82.5, "minimized": 41.2},
    }

    style_config = {
        "technicality": 0.2,
        "verbosity": 0.8,
        "depth": 0.5,
        "perspective": 0.9,
    }

    gen = NarrativeGenerator()
    print(gen.generate(sample_data, style_config))
