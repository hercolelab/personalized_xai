import requests
import json
import yaml

class NarrativeGenerator:
    def __init__(self, model_name="llama3", config_path='src/prompts/prompts.yaml'):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        
        # Load the externalized prompts
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
            self.prompts = full_config['system_prompts']['narrative_generator']

    def _get_dynamic_instructions(self, style):
        """Helper to build style instructions based on numeric thresholds."""
        instr = []
        sf = self.prompts['style_guidelines']['style_features']
        
        # Technicality Logic
        if style['technicality'] < 0.4:
            instr.append(f"TECHNICALITY LOW: {sf['technicality']['low']['description']}")
        elif style['technicality'] > 0.7:
            instr.append(f"TECHNICALITY HIGH: {sf['technicality']['high']['description']}")

        # Verbosity Logic
        if style['verbosity'] < 0.4:
            instr.append(f"VERBOSITY LOW: {sf['verbosity']['low']['description']} {sf['verbosity']['low']['format']}")
        elif style['verbosity'] > 0.7:
            instr.append(f"VERBOSITY HIGH: {sf['verbosity']['high']['description']}")

        # Depth Logic
        if style['depth'] < 0.4:
            instr.append(f"DEPTH LOW: {sf['depth']['low']['description']}")
        elif style['depth'] > 0.7:
            instr.append(f"DEPTH HIGH: {sf['depth']['high']['description']}")

        # Perspective Logic
        if style['perspective'] < 0.4:
            instr.append(f"PERSPECTIVE LOW: {sf['perspective']['low']['description']} Tone: {sf['perspective']['low']['tone']}")
        elif style['perspective'] > 0.7:
            instr.append(f"PERSPECTIVE HIGH: {sf['perspective']['high']['description']} Tone: {sf['perspective']['high']['tone']}")
            
        return "\n".join([f"- {i}" for i in instr])

    def generate(self, raw_data, style, hint=None):
        dynamic_style = self._get_dynamic_instructions(style)
        data_json = json.dumps(raw_data, indent=2)

        # Build the final prompt using YAML content
        prompt = f"""
        [SYSTEM ROLE]
        {self.prompts['role_definition']}

        [CLINICAL GUARDRAILS]
        {chr(10).join(self.prompts['clinical_guardrails'])}

        [DYNAMIC STYLE INSTRUCTIONS]
        {dynamic_style}

        [TAGGING RULES]
        {chr(10).join(self.prompts['tagging_rules'])}

        [REFERENCE EXAMPLES]
        Patient-Style Example: {self.prompts['few_shot_examples']['style_examples']['patient_oriented']['example']}
        Clinician-Style Example: {self.prompts['few_shot_examples']['style_examples']['clinician_oriented']['example']}

        [DATA INPUT]
        {data_json}

        [TASK]
        1. REASONING: List static vs dynamic features based on the 'target' dictionary.
        2. NARRATIVE: Provide the final fluid explanation.
        """

        if hint:
            prompt += f"\n\n[CRITICAL CORRECTION REQUIRED]:\n{hint}"

        payload = {
            "model": self.model_name, 
            "prompt": prompt, 
            "stream": False, 
            "options": {"temperature": 0.2} # Lower temp for better data fidelity
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            res_text = response.json()['response']
            
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
      "shap_impacts": {"Glucose": 0.12, "BMI": 0.08},
      "shap_ranking": ["Glucose", "BMI"],
      "probabilities": {"current": 82.5, "minimized": 41.2}
    }
    
    style_config = {
        "technicality": 0.2, 
        "verbosity": 0.8, 
        "depth": 0.5, 
        "perspective": 0.9
    }
    
    gen = NarrativeGenerator()
    print(gen.generate(sample_data, style_config))