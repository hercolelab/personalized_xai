import requests
import json

class NarrativeGenerator:
    def __init__(self, model_name="llama3"):
        """
        Uses a local Ollama instance to generate personalized medical narratives.
        """
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"

    def generate(self, raw_data, style):
        """
        Generates the explanation based on CPM style vectors.
        Style: {technicality, verbosity, depth, perspective}
        """
        
        # 1. Map CPM Perspective to a specific narrative focus
        # Perspective near 1.0 = Action/DiCE focus; near 0.0 = Cause/SHAP focus
        p = style['perspective']
        if p < 0.4:
            p_instr = "Prioritize the CAUSE of the risk (SHAP values) and why the model flagged the patient."
        elif p > 0.6:
            p_instr = "Prioritize the SOLUTION (Counterfactual Targets) and specific lifestyle changes."
        else:
            p_instr = "Provide a balanced explanation: start with the causes, then transition to actions."

        # 2. Build the System Prompt with CPM constraints
        prompt = f"""
        [SYSTEM ROLE]
        You are an expert Medical XAI Assistant. Your goal is to explain diabetes risk to a patient.
        
        [INPUT DATA]
        {raw_data}
        
        [STYLISTIC CONSTRAINTS]
        - Technicality Level ({style['technicality']}): 0=Simple language, 1=Medical jargon.
        - Verbosity Level ({style['verbosity']}): 0=Extremely brief, 1=Very detailed and conversational.
        - Depth Level ({style['depth']}): 0=Only surface facts, 1=Explain physiological connections.
        - Focus: {p_instr}
        
        [INSTRUCTIONS]
        Write the narrative now. Use only the provided clinical values. 
        If the status is 'Heuristic Fallback', emphasize that these are risk mitigation targets.
        """

        # 3. Request to local Ollama API
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            return response.json()['response']
        except Exception as e:
            return f" [Narrator Error] Local SLM connection failed: {str(e)}"