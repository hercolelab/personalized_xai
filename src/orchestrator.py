import re
from typing import Any, Optional
import yaml
import argparse
import torch
import numpy as np
from termcolor import colored
from src.components.a_setup_xai import XAI_Backend
from src.components.b_cpm import ContextualPreferenceModel
from src.components.c_generator import NarrativeGenerator
from src.components.d_verifiers import XAIVerifier
from src.components.e_feedback_translator import FeedbackTranslator
from src.components.f_refiner import NarrativeRefiner
from src.components.g_rag import RAGRetriever


def clean_tags(text):
    """Removes all XAI tags for the final user-facing output."""
    text = re.sub(r"\[/?V_[^\]]+\]", "", text)
    text = re.sub(r"\[/?I_[^\]]+\]", "", text)
    # return re.sub(r'\s+', ' ', text).strip()
    return text


def _infer_dataset_name(cfg):
    csv_path = cfg.get("data", {}).get("csv_path", "diabetes")
    return csv_path.split("/")[-1].replace("_cleaned.csv", "").replace(".csv", "")


class XAI_Orchestrator:
    def __init__(self, config_path="src/config/config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        print(
            colored(
                "[System] Initializing generalized XAI Pipeline...",
                "white",
                attrs=["bold"],
            )
        )
        self.backend = XAI_Backend(config_path)
        self.cpm = ContextualPreferenceModel(config_path)

        # Initialize RAG if enabled
        self.rag_enabled = self.cfg.get("rag", {}).get("enabled", False)
        self.rag = None
        if self.rag_enabled:
            try:
                dataset_name = _infer_dataset_name(self.cfg)
                self.rag = RAGRetriever(dataset_name, config_path).initialize()
                if self.rag.is_available():
                    print(
                        colored(
                            f"  [RAG] Loaded knowledge base with {self.rag.collection.count()} chunks",
                            "green",
                        )
                    )
                else:
                    print(
                        colored(
                            "  [RAG] No knowledge base available, continuing without RAG",
                            "yellow",
                        )
                    )
                    self.rag = None
            except Exception as e:
                print(colored(f"  [RAG] Failed to initialize: {e}", "yellow"))
                self.rag = None

        self.narrator = NarrativeGenerator(
            model_name=self.cfg["verifier"].get("llm_judge_model", "llama3.1:8b"),
            rag_retriever=self.rag,
        )
        self.verifier = XAIVerifier(config_path)

        # Initialize refiner if enabled
        self.refiner_enabled = self.cfg.get("refiner", {}).get("enabled", True)
        if self.refiner_enabled:
            self.refiner = NarrativeRefiner(
                model_name=self.cfg["verifier"].get("llm_judge_model", "llama3.1:8b")
            )

        self.max_retries = self.cfg.get("orchestrator", {}).get("max_retries", 5)
        self._failure_history: list[str] = []

    def run(self, role, instance_idx):
        """
        Manages the fetch -> generate -> verify -> filter loop.
        """
        # 1. FETCH RAW DATA
        # Returns the raw text summary and the ground_truth dictionary
        raw_xai_text, ground_truth = self.backend.get_explanation(instance_idx)

        # PRINT RAW EXPLAINER OUTPUT
        print(
            colored(
                "\n" + "=" * 20 + " RAW BACKEND EXPLAINER OUTPUT " + "=" * 20, "magenta"
            )
        )
        print(raw_xai_text)
        print(colored("Ground Truth Dictionary:", "magenta"))
        print(ground_truth)
        print(colored("=" * 70 + "\n", "magenta"))

        # 2. SETUP STYLE
        self.cpm.initialize(role)
        target_style = self.cpm.get_state()

        print(
            colored(
                f"[Orchestrator] Persona: {role.upper()} | Target Style: {target_style}",
                "cyan",
            )
        )

        # 3. REJECTION SAMPLING LOOP
        last_narrative = None
        last_failures: Optional[dict[str, Any]] = None
        for attempt in range(self.max_retries):
            print(
                colored(
                    f"\n[Attempt {attempt + 1}/{self.max_retries}] Generating Narrative...",
                    "yellow",
                )
            )

            # A. GENERATION or REFINEMENT
            if attempt == 0:
                candidate_narrative = self.narrator.generate(ground_truth, target_style)
            elif self.refiner_enabled and last_narrative and last_failures:
                print(colored("  [Refiner] Refining previous narrative...", "cyan"))
                candidate_narrative = self.refiner.refine(
                    current_narrative=last_narrative,
                    raw_data=ground_truth,
                    target_style=target_style,
                    failures=last_failures,
                )
            else:
                # Fallback: regenerate (backward compatibility)
                candidate_narrative = self.narrator.generate(ground_truth, target_style)

            # PRINT CANDIDATE NARRATIVE
            print(colored(f"-" * 15 + " CANDIDATE NARRATIVE " + "-" * 15, "yellow"))
            print(candidate_narrative)
            print(colored("-" * 51, "yellow"))

            # B. VERIFICATION
            is_faithful, reason_f = self.verifier.verify_faithfulness(
                candidate_narrative, ground_truth, target_style
            )
            is_complete, reason_c = self.verifier.verify_completeness(
                candidate_narrative, ground_truth, target_style
            )
            is_aligned, style_report = self.verifier.verify_alignment(
                candidate_narrative, target_style
            )

            # C. DECISION
            if all([is_aligned, is_faithful, is_complete]):
                print(
                    colored(
                        f"  [SUCCESS] All checks passed on attempt {attempt + 1}.",
                        "green",
                        attrs=["bold"],
                    )
                )

                # Optionally clean tags for final presentation
                final_text = candidate_narrative
                if self.cfg.get("orchestrator", {}).get("clean_tags_from_final", True):
                    final_text = clean_tags(candidate_narrative)

                return {
                    "status": "success",
                    "narrative": final_text,
                    "attempts": attempt + 1,
                    "style_report": style_report,
                }
            else:
                # Failure Logging
                print(colored(f"  [REJECTED] Validation failed:", "red"))
                if not is_faithful:
                    print(colored(f"    - Faithfulness: {reason_f}", "red"))
                if not is_complete:
                    print(colored(f"    - Completeness: {reason_c}", "red"))
                if not is_aligned:
                    failed_dims = (
                        style_report.get("failed")
                        if isinstance(style_report, dict)
                        else style_report
                    )
                    print(
                        colored(
                            f"    - Alignment: Failed dimensions {failed_dims}",
                            "red",
                        )
                    )

                last_narrative = candidate_narrative
                last_failures = {
                    "faithfulness": {"passed": is_faithful, "reason": reason_f},
                    "completeness": {"passed": is_complete, "reason": reason_c},
                    "alignment": {"passed": is_aligned, "report": style_report},
                }

                self._update_failure_history(last_failures)
                if self._failure_history:
                    last_failures["history"] = list(self._failure_history)

        return {
            "status": "failed",
            "narrative": "Could not verify narrative within retry limit.",
        }

    def _update_failure_history(self, failures: dict[str, Any]):
        if not self.refiner_enabled:
            return
        failure_lines = self.refiner._format_failures(failures)
        for line in failure_lines:
            if line not in self._failure_history:
                self._failure_history.append(line)


class HumanInteractiveOrchestrator:
    def __init__(self, config_path="src/config/config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        print(
            colored(
                "[System] Initializing interactive XAI Pipeline...",
                "white",
                attrs=["bold"],
            )
        )
        self.backend = XAI_Backend(config_path)
        self.cpm = ContextualPreferenceModel(config_path)

        # Initialize RAG if enabled
        self.rag_enabled = self.cfg.get("rag", {}).get("enabled", False)
        self.rag = None
        if self.rag_enabled:
            try:
                dataset_name = _infer_dataset_name(self.cfg)
                self.rag = RAGRetriever(dataset_name, config_path).initialize()
                if self.rag.is_available():
                    print(
                        colored(
                            f"  [RAG] Loaded knowledge base with {self.rag.collection.count()} chunks",
                            "green",
                        )
                    )
                else:
                    print(
                        colored(
                            "  [RAG] No knowledge base available, continuing without RAG",
                            "yellow",
                        )
                    )
                    self.rag = None
            except Exception as e:
                print(colored(f"  [RAG] Failed to initialize: {e}", "yellow"))
                self.rag = None

        self.narrator = NarrativeGenerator(
            model_name=self.cfg["verifier"].get("llm_judge_model", "llama3.1:8b"),
            rag_retriever=self.rag,
        )
        self.verifier = XAIVerifier(config_path)
        self.feedback_translator = FeedbackTranslator(config_path)

        self.refiner_enabled = self.cfg.get("refiner", {}).get("enabled", True)
        if self.refiner_enabled:
            self.refiner = NarrativeRefiner(
                model_name=self.cfg["verifier"].get("llm_judge_model", "llama3.1:8b")
            )

        self.max_retries = self.cfg.get("orchestrator", {}).get("max_retries", 5)
        self._failure_history: list[str] = []

    def _generate_verified(self, ground_truth, target_style):
        last_narrative = None
        last_failures: Optional[dict[str, Any]] = None
        for attempt in range(self.max_retries):
            print(
                colored(
                    f"\n[Attempt {attempt + 1}/{self.max_retries}] Generating Narrative...",
                    "yellow",
                )
            )

            if attempt == 0:
                candidate_narrative = self.narrator.generate(ground_truth, target_style)
            elif self.refiner_enabled and last_narrative and last_failures:
                print(colored("  [Refiner] Refining previous narrative...", "cyan"))
                candidate_narrative = self.refiner.refine(
                    current_narrative=last_narrative,
                    raw_data=ground_truth,
                    target_style=target_style,
                    failures=last_failures,
                )
            else:
                candidate_narrative = self.narrator.generate(ground_truth, target_style)

            print(colored(f"-" * 15 + " CANDIDATE NARRATIVE " + "-" * 15, "yellow"))
            print(candidate_narrative)
            print(colored("-" * 51, "yellow"))

            is_faithful, reason_f = self.verifier.verify_faithfulness(
                candidate_narrative, ground_truth, target_style
            )
            is_complete, reason_c = self.verifier.verify_completeness(
                candidate_narrative, ground_truth, target_style
            )
            is_aligned, style_report = self.verifier.verify_alignment(
                candidate_narrative, target_style
            )

            if all([is_aligned, is_faithful, is_complete]):
                print(
                    colored(
                        f"  [SUCCESS] All checks passed on attempt {attempt + 1}.",
                        "green",
                        attrs=["bold"],
                    )
                )

                final_text = candidate_narrative
                if self.cfg.get("orchestrator", {}).get("clean_tags_from_final", True):
                    final_text = clean_tags(candidate_narrative)

                return {
                    "status": "success",
                    "narrative": final_text,
                    "attempts": attempt + 1,
                    "style_report": style_report,
                }
            else:
                print(colored("  [REJECTED] Validation failed:", "red"))
                if not is_faithful:
                    print(colored(f"    - Faithfulness: {reason_f}", "red"))
                if not is_complete:
                    print(colored(f"    - Completeness: {reason_c}", "red"))
                if not is_aligned:
                    failed_dims = (
                        style_report.get("failed")
                        if isinstance(style_report, dict)
                        else style_report
                    )
                    print(
                        colored(
                            f"    - Alignment: Failed dimensions {failed_dims}",
                            "red",
                        )
                    )

                last_narrative = candidate_narrative
                last_failures = {
                    "faithfulness": {"passed": is_faithful, "reason": reason_f},
                    "completeness": {"passed": is_complete, "reason": reason_c},
                    "alignment": {"passed": is_aligned, "report": style_report},
                }

                self._update_failure_history(last_failures)
                if self._failure_history:
                    last_failures["history"] = list(self._failure_history)

        return {
            "status": "failed",
            "narrative": "Could not verify narrative within retry limit.",
        }

    def _update_failure_history(self, failures: dict[str, Any]):
        if not self.refiner_enabled:
            return
        failure_lines = self.refiner._format_failures(failures)
        for line in failure_lines:
            if line not in self._failure_history:
                self._failure_history.append(line)

    def prepare_case(self, instance_idx):
        raw_xai_text, ground_truth = self.backend.get_explanation(instance_idx)
        print(
            colored(
                "\n" + "=" * 20 + " RAW BACKEND EXPLAINER OUTPUT " + "=" * 20, "magenta"
            )
        )
        print(raw_xai_text)
        print(colored("Ground Truth Dictionary:", "magenta"))
        print(ground_truth)
        print(colored("=" * 70 + "\n", "magenta"))
        return ground_truth

    def reset_style(self):
        self.cpm.reset_to_default()

    def apply_feedback(self, feedback_text: str):
        deltas = self.feedback_translator.translate(feedback_text)
        print(colored(f"[Feedback] Deltas: {deltas}", "cyan"))
        self.cpm.apply_deltas(deltas)

    def run_interactive(self, instance_idx):
        self.reset_style()
        ground_truth = self.prepare_case(instance_idx)

        done_keywords = {"done"}
        while True:
            target_style = self.cpm.get_state()
            print(
                colored(
                    f"[Interactive] Target Style: {target_style}",
                    "cyan",
                )
            )

            results = self._generate_verified(ground_truth, target_style)
            if results["status"] != "success":
                print(
                    colored(
                        "\n[!] Pipeline Failed to produce a safe narrative.",
                        "red",
                        attrs=["bold"],
                    )
                )
                return results

            print(
                "\n"
                + colored(
                    "*" * 30 + " VERIFIED NARRATIVE " + "*" * 30,
                    "green",
                    attrs=["bold"],
                )
            )
            print(results["narrative"])
            print(colored("*" * 80, "green"))

            feedback = input(
                "\nType feedback to refine, or 'done' if satisfied: "
            ).strip()
            if feedback.lower() in done_keywords:
                print(colored("[Interactive] Session complete.", "green"))
                return results

            if not feedback:
                print(colored("[Interactive] No feedback provided.", "yellow"))
                continue

            self.apply_feedback(feedback)


def main():
    parser = argparse.ArgumentParser(description="XAI Orchestrator")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run the human-interactive feedback loop.",
    )
    parser.add_argument(
        "--instance_idx",
        type=int,
        default=None,
        help="Index of instance to explain. Default is highest-risk case.",
    )
    args = parser.parse_args()

    # Load settings
    config_file = "src/config/config.yaml"
    if args.interactive:
        orchestrator = HumanInteractiveOrchestrator(config_file)
    else:
        orchestrator = XAI_Orchestrator(config_file)

    # Pick the most interesting case (highest risk)
    with torch.no_grad():
        X_test_t = torch.tensor(orchestrator.backend.X_test.values, dtype=torch.float32)
        probs = orchestrator.backend.model(X_test_t).numpy().flatten()

    best_idx = int(np.argmax(probs))
    instance_idx = args.instance_idx if args.instance_idx is not None else best_idx

    # Run the loop
    if args.interactive:
        results = orchestrator.run_interactive(instance_idx)
    else:
        results = orchestrator.run("test", instance_idx)

    # FINAL OUTPUT
    if results["status"] == "success":
        print(
            "\n"
            + colored(
                "*" * 30 + " FINAL VERIFIED NARRATIVE " + "*" * 30,
                "green",
                attrs=["bold"],
            )
        )
        print(results["narrative"])
        print(colored("*" * 86, "green"))
    else:
        print(
            colored(
                "\n[!] Pipeline Failed to produce a safe narrative.",
                "red",
                attrs=["bold"],
            )
        )


if __name__ == "__main__":
    main()
