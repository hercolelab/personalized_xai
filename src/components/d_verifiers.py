import re
import json
import requests
import numpy as np
import yaml
import os
from dotenv import load_dotenv

# Load environment variables from .env if present (project root or current dir)
load_dotenv()

class XAIVerifier:
    def __init__(self, config_path='src/config/config.yaml', prompt_path='src/prompts/prompts.yaml'):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        with open(prompt_path, 'r') as f:
            full_prompts = yaml.safe_load(f)
            try:
                self.style_defs = full_prompts['system_prompts']['narrative_generator']['style_guidelines']['style_features']
            except KeyError as e:
                print(f"[Verifier Error] Path mismatch in prompts.yaml: {e}")
                raise

        self.model_name = self.cfg['verifier'].get('llm_judge_model', 'llama3')
        
        # Determine if using cloud model and set API URL accordingly
        is_cloud_model = '-cloud' in self.model_name or ':cloud' in self.model_name
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
        
        self.tolerances = self.cfg['verifier'].get('tolerances', {})
        self.alias_map = self.cfg['verifier'].get('alias_map', {})

    # --- 1. FAITHFULNESS VERIFICATION (Values match ground truth) ---
    def verify_faithfulness(self, narrative, ground_truth, style):
        """
        Ensures that whenever a value is mentioned in tags, it matches the XAI Backend output.
        Does not check presence of data; that is handled by completeness.
        """
        # Allow ASCII hyphen and Unicode minus/dash (U+2011 nbh, U+2013 en dash, U+2212 minus)
        _num = r"[\d.\-\u2011\u2013\u2212]+"
        curr_regex = re.findall(rf"\[V_START_(\w+)\]({_num})\[/V_START_\w+\]", narrative)
        targ_regex = re.findall(rf"\[V_GOAL_(\w+)\]({_num})\[/V_GOAL_\w+\]", narrative)
        imp_regex = re.findall(rf"\[I_(\w+)\]({_num})\[/I_\w+\]", narrative)

        def _norm(s):
            if not s:
                return s
            for u in ("\u2011", "\u2013", "\u2212"):
                s = s.replace(u, "-")
            return s

        extracted = {
            "current": {k.upper(): _norm(v) for k, v in curr_regex},
            "target": {k.upper(): _norm(v) for k, v in targ_regex},
            "impacts": {k.upper(): _norm(v) for k, v in imp_regex}
        }

        # Numerical Validation: every mentioned value must match ground truth
        def validate_values(extracted_dict, gt_dict, label_type):
            for feat, val in extracted_dict.items():
                try:
                    num_val = float(val)
                    match = next((k for k in gt_dict if k.upper() in feat), None)
                    if match:
                        actual = float(gt_dict[match])
                        tol = self.tolerances.get('impact' if label_type == "IMPACT" else 'default', 0.5)
                        if abs(num_val - actual) > tol:
                            return False, f"{label_type} mismatch: {feat} is {num_val}, expected {actual}"
                except (ValueError, TypeError): continue
            return True, ""

        for cat, lbl in [("current", "START"), ("target", "GOAL"), ("impacts", "IMPACT")]:
            gt_key = "shap_impacts" if cat == "impacts" else cat
            ok, err = validate_values(extracted[cat], ground_truth.get(gt_key, {}), lbl)
            if not ok: return False, err

        return True, "Faithfulness Verified"
    
    # --- 2. COMPLETENESS VERIFICATION (All required data present) ---
    def verify_completeness(self, narrative, ground_truth, style):
        """
        Checks that all required data are present (not that values are correct).
        - Perspective: required features to mention (changes only vs changes + top SHAP).
        - High technicality/depth (>0.7): SHAP values must be present for all target features.
        """
        # Identify features that actually changed significantly (Counterfactuals)
        actual_changes = []
        for feat, target_val in ground_truth.get('target', {}).items():
            current_val = ground_truth.get('current', {}).get(feat)
            if current_val is not None:
                if abs(float(target_val) - float(current_val)) > 0.1:
                    actual_changes.append(feat)

        # Require only features that are actually in the counterfactual (target), not other top-SHAP features
        required_features = set(actual_changes)
        if style['perspective'] >= 0.5:
            mode_label = "Proactive (Changes only)"
        else:
            mode_label = "Retroactive (Changes + SHAP Drivers)"

        missing = set()
        for feat in required_features:
            feat_upper = feat.upper()
            valid_names = [feat_upper] + self.alias_map.get(feat_upper, [])
            pattern = "|".join([re.escape(n) for n in valid_names])
            found = re.search(rf"\[(V_START|V_GOAL|I)_({pattern})\]", narrative, re.IGNORECASE) or \
                    any(name.lower() in narrative.lower() for name in valid_names)
            if not found:
                missing.add(feat)

        # When technicality or depth is high, require SHAP values present for all target features
        requires_shap = style.get('technicality', 0) > 0.7 or style.get('depth', 0) > 0.7
        if requires_shap:
            target_features = list(ground_truth.get('target', {}).keys())
            for feat in target_features:
                feat_upper = feat.upper()
                valid_names = [feat_upper] + self.alias_map.get(feat_upper, [])
                pattern = "|".join([re.escape(n) for n in valid_names])
                # Require [I_FEATURE]value[/I_...] to be present (allow Unicode minus/dash in value)
                _num = r"[\d.\-\u2011\u2013\u2212]+"
                shap_present = re.search(rf"\[I_({pattern})\]({_num})\[/I_\w+\]", narrative, re.IGNORECASE)
                if not shap_present:
                    missing.add(feat)
            if target_features:
                mode_label = f"{mode_label}; SHAP required for target features"

        if missing:
            return False, f"Completeness Error [{mode_label}]: Missing data for {sorted(missing)}."
        
        return True, f"Completeness Verified ({mode_label})"

    # --- 3. ALIGNMENT VERIFICATION (LLM-as-a-Judge) ---
    def verify_alignment(self, narrative, target_style_dict):
        rubric_str = ""
        available_levels = []
        for dim, config in self.cfg['verifier']['verifier_rubric'].items():
            levels = "\n".join([f"- {val}: {desc}" for val, desc in config['levels'].items()])
            rubric_str += f"\nDimension: {dim.upper()}\n{levels}"
            available_levels.extend(config['levels'].keys())

        unique_levels = sorted(list(set(available_levels)))

        judge_prompt = f""" You are a helpful assistant that evaluates the stylistic profile of a narrative with the provided rubric guidelines.
        The narrative is a description of a counterfactual explanation, supported by SHAP feature attributions.
        
        # TASK
        Your task is:
        1. Read the narrative and the rubric guidelines.
        2. Evaluate the narrative by selecting the closest numerical level from this list: {unique_levels}.

        # OUTPUT FORMAT
        Return ONLY a JSON object with the following keys: {{"technicality": <number>, "verbosity": <number>, "depth": <number>, "perspective": <number>}}.
        No other text or comments.
        
        # RUBRIC GUIDELINES
        {rubric_str}

        Here is the narrative to evaluate:
        # NARRATIVE
        {narrative}
        """

        try:
            res = self._query_llm(judge_prompt)
            scores = json.loads(re.search(r'\{.*\}', res, re.DOTALL).group())
            
            dims = ['technicality', 'verbosity', 'depth', 'perspective']
            p_vec = np.array([scores.get(d, 0.5) for d in dims])
            t_vec = np.array([target_style_dict[d] for d in dims])

            diffs = np.abs(p_vec - t_vec)
            tolerance = self.tolerances.get('alignment', 0.35)
            
            failed = [dims[i] for i, d in enumerate(diffs) if d > tolerance]
            return len(failed) == 0, {"perceived": scores, "target": target_style_dict, "failed": failed}
        except Exception as e:
            return False, {"error": f"Judge failure: {str(e)}"}

    def _query_llm(self, prompt):
        payload = {"model": self.model_name, "prompt": prompt, "stream": False, "format": "json"}
        try:
            headers = {}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            response = requests.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()['response']
        except:
            return "{}"