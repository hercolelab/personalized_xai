import json
import yaml

from src.components.ollama_client import OllamaClient


class NarrativeGenerator:
    def __init__(
        self,
        model_name="gpt-oss:20b-cloud",
        config_path="src/prompts/prompts.yaml",
        rag_retriever=None,
    ):
        """
        Initialize the NarrativeGenerator.

        Args:
            model_name: Ollama model name for generation
            config_path: Path to prompts YAML config
            rag_retriever: Optional RAGRetriever instance for context augmentation
        """
        self.model_name = model_name
        self.ollama = OllamaClient.from_model(self.model_name)
        self.rag = rag_retriever

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

    def _build_rag_query(self, raw_data: dict) -> str:
        """
        Build a query string from raw explanation data for RAG retrieval.

        Args:
            raw_data: The raw explanation data dict

        Returns:
            A query string suitable for semantic search
        """

        current = raw_data.get("current", {})
        target = raw_data.get("target", {})

        changed_features = list(target.keys())

        query_parts = [
            f"Explain the health implications of {', '.join(changed_features)}."
        ]

        for feat in changed_features:
            if feat in current:
                query_parts.append(
                    f"{feat}: changing from {current[feat]} to {target[feat]}"
                )

        return " ".join(query_parts)

    def generate(self, raw_data, style, hint=None, use_rag: bool = True):
        """
        Generate a narrative explanation.

        Args:
            raw_data: The raw explanation data
            style: Style configuration dict
            hint: Optional hint for corrections
            use_rag: Whether to use RAG context (if available)

        Returns:
            Generated narrative string
        """
        dynamic_style = self._get_dynamic_instructions(style)
        data_json = json.dumps(raw_data, indent=2)

        rag_context = ""
        if use_rag and self.rag and self.rag.is_available():
            query = self._build_rag_query(raw_data)
            print(f" [RAG] Query: {query}")
            chunks = self.rag.retrieve_with_metadata(query)
            print(f" [RAG] Retrieved {len(chunks)} chunks")
            if chunks:
                for i, c in enumerate(chunks):
                    print(f"   [{i + 1}] sim={c['similarity']:.3f} from {c['source']}")
                rag_context = (
                    "\n\n# DOMAIN KNOWLEDGE (Use this to enhance your explanations)\n"
                )
                rag_context += "\n---\n".join([c["text"] for c in chunks])
                print(f" [RAG] Context: {rag_context}")

        prompt_template = self.prompts["prompt_template"]
        format_kwargs = {"data_json": data_json, "dynamic_style": dynamic_style}
        if "{rag_context}" in prompt_template:
            format_kwargs["rag_context"] = rag_context
        prompt = prompt_template.format(**format_kwargs)

        if rag_context and "{rag_context}" not in prompt_template:
            prompt += rag_context

        if hint:
            prompt += f"\n\n# CRITICAL CORRECTION REQUIRED:\n{hint}"

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
