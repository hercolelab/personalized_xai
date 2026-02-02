import json
import yaml
from src.components.ollama_client import OllamaClient


class NarrativeRefiner:
    """
    A component that refines failed narratives based on structured verification failures.
    """

    def __init__(
        self, model_name="gpt-oss:20b-cloud", config_path="src/prompts/prompts.yaml"
    ):
        self.model_name = model_name
        self.ollama = OllamaClient.from_model(self.model_name)

        with open(config_path, "r") as f:
            full_config = yaml.safe_load(f)
            self.prompt_template = full_config["system_prompts"]["narrative_refiner"][
                "prompt"
            ]

    def _format_failures(self, failures: dict) -> str:
        """Format the failures dictionary into a readable string for the LLM."""
        failure_lines = []

        # Faithfulness failures
        if not failures.get("faithfulness", {}).get("passed", True):
            reason = failures["faithfulness"].get(
                "reason", "Unknown faithfulness issue"
            )
            failure_lines.append(f"FAITHFULNESS ERROR: {reason}")

        # Completeness failures
        if not failures.get("completeness", {}).get("passed", True):
            reason = failures["completeness"].get(
                "reason", "Unknown completeness issue"
            )
            failure_lines.append(f"COMPLETENESS ERROR: {reason}")

        # Alignment failures
        if not failures.get("alignment", {}).get("passed", True):
            report = failures["alignment"].get("report", {})
            if isinstance(report, dict):
                failed_dims = report.get("failed", [])
                perceived = report.get("perceived", {})
                target = report.get("target", {})
                if failed_dims:
                    for dim in failed_dims:
                        p_val = perceived.get(dim, "?")
                        t_val = target.get(dim, "?")
                        failure_lines.append(
                            f"ALIGNMENT ERROR on '{dim}': perceived={p_val}, target={t_val}"
                        )
            else:
                failure_lines.append(f"ALIGNMENT ERROR: {report}")

        return (
            "\n".join(failure_lines)
            if failure_lines
            else "No specific failures listed."
        )

    def _get_style_instructions(self, style: dict) -> str:
        """Convert style dictionary to human-readable instructions."""
        instructions = []

        # Technicality
        if style["technicality"] < 0.4:
            instructions.append(
                "TECHNICALITY: Use plain, accessible language. Avoid jargon. Do NOT use SHAP values."
            )
        elif style["technicality"] > 0.7:
            instructions.append(
                "TECHNICALITY: Use clinical/technical language. Include SHAP values with precise measurements."
            )
        else:
            instructions.append(
                "TECHNICALITY: Use balanced language with some technical terms. Do NOT use SHAP values."
            )

        # Verbosity
        if style["verbosity"] < 0.4:
            instructions.append(
                "VERBOSITY: Be concise. Use bullet points. Short sentences only."
            )
        elif style["verbosity"] > 0.7:
            instructions.append(
                "VERBOSITY: Use flowing narrative paragraphs. NO bullet points. Use transition words."
            )
        else:
            instructions.append(
                "VERBOSITY: Use standard paragraphs or mixed formatting."
            )

        # Depth
        if style["depth"] < 0.4:
            instructions.append(
                "DEPTH: Analyze features in isolation (one by one). Do not explain interactions."
            )
        elif style["depth"] > 0.7:
            instructions.append(
                "DEPTH: Provide deep systemic analysis. Explain how features interact with each other."
            )
        else:
            instructions.append(
                "DEPTH: Acknowledge multiple features contribute but avoid detailed interaction analysis."
            )

        # Perspective
        if style["perspective"] < 0.4:
            instructions.append(
                "PERSPECTIVE: Be descriptive/retroactive. Explain the current situation."
            )
        elif style["perspective"] > 0.7:
            instructions.append(
                "PERSPECTIVE: Be proactive/coaching. Frame as action plan with future-oriented guidance."
            )
        else:
            instructions.append(
                "PERSPECTIVE: Balance descriptive reporting with some forward-looking statements."
            )

        return "\n".join(instructions)

    def refine(
        self,
        current_narrative: str,
        raw_data: dict,
        target_style: dict,
        failures: dict,
    ) -> str:
        """
        Refines a failed narrative based on structured verification failures.

        Args:
            current_narrative: The narrative that failed verification
            raw_data: The ground truth explanation data (current, target, shap_impacts, probabilities)
            target_style: The target style dictionary with technicality, verbosity, depth, perspective
            failures: Dictionary containing structured failure information:
                - faithfulness: {"passed": bool, "reason": str}
                - completeness: {"passed": bool, "reason": str}
                - alignment: {"passed": bool, "report": dict}

        Returns:
            A refined narrative addressing all failures
        """

        formatted_failures = self._format_failures(failures)
        style_instructions = self._get_style_instructions(target_style)
        data_json = json.dumps(raw_data, indent=2)

        prompt = self.prompt_template.replace("{{narrative}}", current_narrative)
        prompt = prompt.replace("{{raw_data}}", data_json)
        prompt = prompt.replace("{{target_style}}", style_instructions)
        prompt = prompt.replace("{{failures}}", formatted_failures)

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }

        try:
            response = self.ollama.post(payload)
            response.raise_for_status()
            res_text = response.json()["response"]

            if "NARRATIVE:" in res_text:
                return res_text.split("NARRATIVE:")[1].strip()
            return res_text.strip()
        except Exception as e:
            print(f"[Refiner] Error during refinement: {str(e)}")
            return current_narrative


if __name__ == "__main__":
    sample_narrative = """
    Your glucose level is currently [V_START_GLUCOSE]189[/V_START_GLUCOSE] and needs to 
    decrease to [V_GOAL_GLUCOSE]110[/V_GOAL_GLUCOSE].
    """

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

    sample_style = {
        "technicality": 0.2,
        "verbosity": 0.8,
        "depth": 0.5,
        "perspective": 0.9,
    }

    sample_failures = {
        "faithfulness": {
            "passed": False,
            "reason": "Glucose START mismatch: 189 vs 148",
        },
        "completeness": {"passed": True, "reason": ""},
        "alignment": {"passed": True, "report": {}},
    }

    refiner = NarrativeRefiner()
    refined = refiner.refine(
        sample_narrative, sample_data, sample_style, sample_failures
    )
    print("Refined narrative:")
    print(refined)
