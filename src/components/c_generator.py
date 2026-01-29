import requests
import json
import yaml
import os
from dotenv import load_dotenv

# Load environment variables from .env if present (project root or current dir)
load_dotenv()

class NarrativeGenerator:
    def __init__(self, model_name="gpt-oss:20b-cloud", config_path='src/prompts/prompts.yaml'):
        self.model_name = model_name
        
        # Determine if using cloud model and set API URL accordingly
        is_cloud_model = '-cloud' in model_name or ':cloud' in model_name
        if is_cloud_model:
            self.api_url = "https://ollama.com/api/generate"
            # Get API key from environment variable
            self.api_key = os.getenv('OLLAMA_API_KEY')
            if not self.api_key:
                print("[Warning] OLLAMA_API_KEY not set. Cloud models require authentication.")
                print("[Info] Set OLLAMA_API_KEY environment variable or run 'ollama signin'")
        else:
            self.api_url = "http://localhost:11434/api/generate"
            self.api_key = None
        
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
        else:
            instr.append(f"TECHNICALITY MEDIUM: {sf['technicality']['medium']['description']}")

        # Verbosity Logic
        if style['verbosity'] < 0.4:
            instr.append(f"VERBOSITY LOW: {sf['verbosity']['low']['description']}")
        elif style['verbosity'] > 0.7:
            instr.append(f"VERBOSITY HIGH: {sf['verbosity']['high']['description']}")
        else:
            instr.append(f"VERBOSITY MEDIUM: {sf['verbosity']['medium']['description']}")

        # Depth Logic
        if style['depth'] < 0.4:
            instr.append(f"DEPTH LOW: {sf['depth']['low']['description']}")
        elif style['depth'] > 0.7:
            instr.append(f"DEPTH HIGH: {sf['depth']['high']['description']}")
        else:
            instr.append(f"DEPTH MEDIUM: {sf['depth']['medium']['description']}")

        # Perspective Logic
        if style['perspective'] < 0.4:
            instr.append(f"PERSPECTIVE LOW: {sf['perspective']['low']['description']} Tone: {sf['perspective']['low']['tone']}")
        elif style['perspective'] > 0.7:
            instr.append(f"PERSPECTIVE HIGH: {sf['perspective']['high']['description']} Tone: {sf['perspective']['high']['tone']}")
        else:
            instr.append(f"PERSPECTIVE MEDIUM: {sf['perspective']['medium']['description']} Tone: {sf['perspective']['medium']['tone']}")
            
        return "\n".join([f"- {i}" for i in instr])

    def generate(self, raw_data, style, hint=None):
        dynamic_style = self._get_dynamic_instructions(style)
        data_json = json.dumps(raw_data, indent=2)

        # Build the final prompt using YAML content
        prompt = f"""You are an expert XAI (Explainable AI) Interpreter and Data Narrator. Your role is to bridge the gap between complex machine learning algorithms and human understanding. You translate mathematical outputs into clear, actionable natural language narratives.
        
        To perform this task, you must understand two core concepts:
        1.  Counterfactual Explanation: A "What-if" scenario that identifies the minimal changes required in the user's current situation to achieve a different, desired model outcome (e.g., flipping a loan rejection to approval). It focuses on actionability.
        2.  SHAP (SHapley Additive exPlanations): A game-theoretic method that explains a machine-learning model’s prediction by assigning each feature a Shapley-based attribution that quantifies its average marginal contribution—positive or negative—to the deviation from the model’s expected output.

        # INPUT DATA STRUCTURE 
        You will receive a JSON object containing the following keys:
        - `current`: A dictionary representing the original profile or situation of the instance.
        - `target`: A dictionary representing the specific list of changes suggested by the counterfactual explanation algorithm.
        - `shap_impacts`: A dictionary containing SHAP values for all features, ordered by magnitude (absolute value) from highest to lowest. Each key is a feature name and each value is the SHAP impact value for that feature.
        - `probabilities`: A dictionary containing:
            - `current`: The prediction probability of the original profile.
            - `minimized`: The prediction probability achieved after applying the counterfactual changes.
        
        # TASK
        Your task is:
        - Write a narrative that describes a Counterfactual Explanation, supported by SHAP Feature Attributions. Explain the journey from the Current state to the Target state. Do not simply list the changes. 
        - Analyze the contribution of the changed features and highlight how the interactions between these features drive the shift in the prediction probability. 
        - Explain why the suggested changes lead to the desired outcome (using the provided SHAP values as evidence of impact if the style allows it).
        
        # STYLE REQUIREMENTS
        The narrative must follow a specific stylistic profile defined along four dimensions:
        - Technicality: ranges from layperson-friendly language with minimal numbers to highly clinical language with precise measurements and technical terms.
        - Verbosity: ranges from very concise, bullet-style communication to more elaborate, flowing narrative text.
        - Depth: ranges from listing individual factors in isolation to explaining how multiple factors interact and jointly affect the outcome.
        - Perspective: ranges from purely descriptive/retroactive reporting of the current situation to proactive, coaching-oriented guidance focused on future changes.

        # FORMAT & TAGGING RULES (CRITICAL)
        To allow for automated post-processing and validation, you must wrap all numerical values and feature names in specific XML-style tags. DO NOT deviate from this format.
        1.  Current Values: When mentioning a value from the user's original profile, wrap it as: `[V_START_FEATURE_NAME]value[/V_START_FEATURE_NAME]`
        2.  Target Values: When mentioning a suggested new value from the counterfactual, wrap it as: `[V_GOAL_FEATURE_NAME]value[/V_GOAL_FEATURE_NAME]`
        3.  SHAP Impacts: When mentioning a SHAP contribution value, wrap it as: `[I_FEATURE_NAME]value[/I_FEATURE_NAME]`

        # CLINICAL GUARDRAILS
        - Static vs Dynamic: Only features in the 'target' dictionary are changeable. Treat all others as static context (NEVER suggest changing them)."
        - SHAP Integration: SHAP values can be used to support the description of the counterfactual explanation, but only if the style allows it. If a feature is not in the 'target' dictionary, DO NOT mention its SHAP value in the narrative.
        - SHAP Suppression: If perspective is high, technicality is low, or verbosity is low, DO NOT mention SHAP impact values.
        
        # ADDITIONAL INSTRUCTIONS
        - Highlight the changes: Do not list the data input entirely, but rather highlight the changes and the impact of the changes.
        - You are the Narrator: Consider the counterfactual and SHAP values not as independent entities, but as a cohesive unit that drives the narrative.
        - Don't use "Counterfactual" term: Instead, use "Suggested Changes" or "Recommended Changes".

        
        Here is your input:
        # DATA INPUT
        {data_json}

        Here is your style instructions:
        # STYLE INSTRUCTIONS
        {dynamic_style}
        """


        if hint:
            prompt += f"\n\n# CRITICAL CORRECTION REQUIRED:\n{hint}"

        payload = {
            "model": self.model_name, 
            "prompt": prompt, 
            "stream": False, 
            "options": {"temperature": 0.2} # Lower temp for better data fidelity
        }
        
        try:
            headers = {}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            response = requests.post(self.api_url, json=payload, headers=headers)
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
      "shap_impacts": {"Glucose": 0.12, "BMI": 0.08, "Age": 0.05, "BloodPressure": 0.03},
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