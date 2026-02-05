import re
import json
import numpy as np
import yaml
from src.components.ollama_client import OllamaClient


class XAIVerifier:
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
            dataset_name = csv_path.split("/")[-1].replace("_cleaned.csv", "").replace(".csv", "")
            prompt_path = f"src/prompts/prompts_{dataset_name}.yaml"

        with open(prompt_path, "r") as f:
            full_prompts = yaml.safe_load(f)
            try:
                self.style_defs = full_prompts["system_prompts"]["narrative_generator"][
                    "style_guidelines"
                ]["style_features"]
            except KeyError as e:
                print(f"[Verifier Error] Path mismatch in {prompt_path}: {e}")
                raise

        self.model_name = self.cfg["verifier"].get("llm_judge_model", "llama3")
        self.ollama = OllamaClient.from_model(self.model_name)

        self.tolerances = self.cfg["verifier"].get("tolerances", {})
        self.alias_map = self.cfg["verifier"].get("alias_map", {})

    # --- 1. FAITHFULNESS VERIFICATION (Values match ground truth) ---
    def verify_faithfulness(self, narrative, ground_truth, style):
        """
        Ensures that whenever a value is mentioned in tags, it matches the XAI Backend output.
        Also validates direction tags when technicality is low.
        Does not check presence of data; that is handled by completeness.
        """
        # Allow ASCII hyphen and Unicode minus/dash (U+2011 nbh, U+2013 en dash, U+2212 minus)
        _num = r"[\d.\-\u2011\u2013\u2212]+"
        curr_regex = re.findall(
            rf"\[V_START_(\w+)\]({_num})\[/V_START_\w+\]", narrative
        )
        targ_regex = re.findall(rf"\[V_GOAL_(\w+)\]({_num})\[/V_GOAL_\w+\]", narrative)
        imp_regex = re.findall(rf"\[I_(\w+)\]({_num})\[/I_\w+\]", narrative)
        # Extract direction tags: [D_FEATURE_NAME]direction[/D_FEATURE_NAME]
        dir_regex = re.findall(rf"\[D_(\w+)\]([^\[]+)\[/D_\w+\]", narrative)

        def _norm(s):
            if not s:
                return s
            for u in ("\u2011", "\u2013", "\u2212"):
                s = s.replace(u, "-")
            return s

        extracted = {
            "current": {k.upper(): _norm(v) for k, v in curr_regex},
            "target": {k.upper(): _norm(v) for k, v in targ_regex},
            "impacts": {k.upper(): _norm(v) for k, v in imp_regex},
            "directions": {k.upper(): v.strip() for k, v in dir_regex},
        }

        # Numerical Validation: every mentioned value must match ground truth
        def validate_values(extracted_dict, gt_dict, label_type):
            for feat, val in extracted_dict.items():
                try:
                    num_val = float(val)
                    match = next((k for k in gt_dict if k.upper() in feat), None)
                    if match:
                        actual = float(gt_dict[match])
                        tol = self.tolerances.get(
                            "impact" if label_type == "IMPACT" else "default", 0.5
                        )
                        if abs(num_val - actual) > tol:
                            return (
                                False,
                                f"{label_type} mismatch: {feat} is {num_val}, expected {actual}",
                            )
                except (ValueError, TypeError):
                    continue
            return True, ""

        for cat, lbl in [
            ("current", "START"),
            ("target", "GOAL"),
            ("impacts", "IMPACT"),
        ]:
            gt_key = "shap_impacts" if cat == "impacts" else cat
            ok, err = validate_values(extracted[cat], ground_truth.get(gt_key, {}), lbl)
            if not ok:
                return False, err

        # Validate directions when technicality is low
        technicality_low = style.get("technicality", 0.5) <= 0.33
        if technicality_low:
            # Validate all direction tags that are present
            for feat, direction in extracted["directions"].items():
                # Find matching feature in ground truth
                match = next((k for k in ground_truth.get("target", {}) if k.upper() in feat), None)
                if match:
                    current_val = ground_truth.get("current", {}).get(match)
                    target_val = ground_truth.get("target", {}).get(match)
                    if current_val is not None and target_val is not None:
                        # Validate direction using LLM
                        is_valid, err = self._validate_direction(
                            current_val, target_val, direction, feat
                        )
                        if not is_valid:
                            return False, f"Direction validation failed for {feat}: {err}"

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
        for feat, target_val in ground_truth.get("target", {}).items():
            current_val = ground_truth.get("current", {}).get(feat)
            if current_val is not None:
                if abs(float(target_val) - float(current_val)) > 0.1:
                    actual_changes.append(feat)

        # Require only features that are actually in the counterfactual (target), not other top-SHAP features
        required_features = set(actual_changes)
        

        missing = set()
        technicality_low = style.get("technicality", 0.5) <= 0.33
        
        for feat in required_features:
            feat_upper = feat.upper()
            valid_names = [feat_upper] + self.alias_map.get(feat_upper, [])
            pattern = "|".join([re.escape(n) for n in valid_names])
            
            # When technicality is low, we don't require V_START, but we require D_ tags
            if technicality_low:
                # Check for direction tag or feature mention
                found = re.search(
                    rf"\[D_({pattern})\]", narrative, re.IGNORECASE
                ) or any(name.lower() in narrative.lower() for name in valid_names)
            else:
                # When technicality is not low, require V_START, V_GOAL, or I tags
                found = re.search(
                    rf"\[(V_START|V_GOAL|I)_({pattern})\]", narrative, re.IGNORECASE
                ) or any(name.lower() in narrative.lower() for name in valid_names)
            
            if not found:
                missing.add(feat)

        # When technicality or depth is high, require SHAP values present for all target features
        requires_shap = (
            style.get("technicality", 0) > 0.7
        )
        if requires_shap:
            target_features = list(ground_truth.get("target", {}).keys())
            for feat in target_features:
                feat_upper = feat.upper()
                valid_names = [feat_upper] + self.alias_map.get(feat_upper, [])
                pattern = "|".join([re.escape(n) for n in valid_names])
                # Require [I_FEATURE]value[/I_...] to be present (allow Unicode minus/dash in value)
                _num = r"[\d.\-\u2011\u2013\u2212]+"
                shap_present = re.search(
                    rf"\[I_({pattern})\]({_num})\[/I_\w+\]", narrative, re.IGNORECASE
                )
                if not shap_present:
                    missing.add(feat)
        
        # When technicality is low, ensure direction tags are present for all changed features
        if technicality_low:
            for feat in required_features:
                feat_upper = feat.upper()
                valid_names = [feat_upper] + self.alias_map.get(feat_upper, [])
                pattern = "|".join([re.escape(n) for n in valid_names])
                direction_present = re.search(
                        rf"\[D_({pattern})\]", narrative, re.IGNORECASE
                )
                if not direction_present:
                    missing.add(f"{feat} (direction tag required)")

        if missing:
            return (
                False,
                f"Completeness Error: Missing data for {sorted(missing)}.",
            )

        return True, f"Completeness Verified"

    # --- 3. ALIGNMENT VERIFICATION (LLM-as-a-Judge) ---
    def verify_alignment(self, narrative, target_style_dict):
        rubric_str = ""
        available_levels = []
        for dim, config in self.cfg["verifier"]["verifier_rubric"].items():
            levels = "\n".join(
                [f"- {val}: {desc}" for val, desc in config["levels"].items()]
            )
            rubric_str += f"\nDimension: {dim.upper()}\n{levels}"
            available_levels.extend(config["levels"].keys())

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
            scores = json.loads(re.search(r"\{.*\}", res, re.DOTALL).group())

            dims = ["technicality", "verbosity", "depth", "perspective"]
            p_vec = np.array([scores.get(d, 0.5) for d in dims])
            t_vec = np.array([target_style_dict[d] for d in dims])

            diffs = np.abs(p_vec - t_vec)
            tolerance = self.tolerances.get("alignment", 0.35)

            failed = [dims[i] for i, d in enumerate(diffs) if d > tolerance]
            return len(failed) == 0, {
                "perceived": scores,
                "target": target_style_dict,
                "failed": failed,
            }
        except Exception as e:
            return False, {"error": f"Judge failure: {str(e)}"}

    def _validate_direction(self, initial_value, changed_value, direction, feature_name):
        """
        Validates that the direction verb correctly describes the change from initial to changed value.
        Uses LLM similar to verify_alignment pattern.
        Returns (is_valid, error_message)
        """
        try:
            initial = float(initial_value)
            changed = float(changed_value)
            
            validation_prompt = f"""You are a helpful assistant that validates whether a direction verb correctly describes a numerical change.

                Given:
                - Feature: {feature_name}
                - Initial value: {initial}
                - Changed value: {changed}
                - Direction verb: "{direction}"

                Task: Determine if the direction verb correctly describes the change from initial to changed value.

                # OUTPUT FORMAT
                Return ONLY a JSON object with this exact structure: {{"final_answer": "Yes"}} or {{"final_answer": "No"}}
                No other text or comments."""

            try:
                res = self._query_llm(validation_prompt)
                result = json.loads(re.search(r"\{.*\}", res, re.DOTALL).group())
                if result.get("final_answer", "").lower() == "no":
                    return False, f"Direction '{direction}' does not match change from {initial} to {changed}"
                return True, ""
            except Exception as e:
                return False, f"Direction validation error: {str(e)}"
        except (ValueError, TypeError) as e:
            return False, f"Invalid values for direction validation: {str(e)}"

    def _query_llm(self, prompt):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }
        try:
            response = self.ollama.post(payload)
            response.raise_for_status()
            return response.json()["response"]
        except:
            return "{}"
