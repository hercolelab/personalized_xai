import argparse
import csv
import json
import os
from typing import TypedDict

import numpy as np
import torch
import yaml
from langgraph.graph import END, StateGraph

from src.components.llm_client import LLMClient
from src.orchestrator import HumanInteractiveOrchestrator


class PersonaFeedbackAgent:
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
            self.prompt_template = prompts["system_prompts"]["persona_feedback"][
                "prompt"
            ]

        self.model_name = self.cfg["verifier"].get("llm_judge_model", "llama3")
        self.ollama = LLMClient.from_model(self.model_name)

    def generate(self, persona_name: str, persona_vector: list, narrative: str) -> str:
        prompt = (
            self.prompt_template.replace("{{persona_name}}", persona_name)
            .replace("{{persona_vector}}", json.dumps(persona_vector))
            .replace("{{narrative}}", narrative)
        )
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        try:
            response = self.ollama.post(payload)
            response.raise_for_status()
            res_text = response.json().get("response", "").strip()
        except Exception:
            res_text = ""

        if not res_text:
            return "Please adjust the explanation to better match my preferences."
        return res_text


class EvalState(TypedDict):
    step: int
    max_steps: int
    done: bool
    should_finalize: bool
    narrative: str
    final_narrative: str
    feedback: str
    error: str


class AgentJudgeEvaluator:
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

        self.orchestrator = HumanInteractiveOrchestrator(config_path)
        self.feedback_agent = PersonaFeedbackAgent(config_path, prompt_path)
        self.alignment_tol = self.cfg.get("orchestrator", {}).get(
            "eval_alignment_tolerance",
            self.cfg["verifier"].get("tolerances", {}).get("alignment", 0.35),
        )

    def _is_satisfied(self, target_vector: np.ndarray) -> bool:
        diffs = np.abs(self.orchestrator.cpm.vector - target_vector)
        return bool(np.all(diffs <= self.alignment_tol))

    def _build_graph(self, persona_name: str, target_vector: np.ndarray, ground_truth):
        def generate_narrative(state: EvalState):
            if state["done"] or state.get("error"):
                return {}

            print(
                f"\n[Eval] Step {state['step']}/{state['max_steps']} | CPM: {self.orchestrator.cpm.get_state()}"
            )
            target_style = self.orchestrator.cpm.get_state()
            results = self.orchestrator._generate_verified(ground_truth, target_style)
            if results["status"] != "success":
                print(
                    f"[Eval] ERROR: Generation failed - {results.get('narrative', 'unknown error')}"
                )
                return {
                    "done": True,
                    "error": "generation_failed",
                }
            return {"narrative": results["narrative"]}

        def persona_feedback(state: EvalState):
            if state["done"] or state.get("error"):
                return {}

            feedback = self.feedback_agent.generate(
                persona_name, target_vector.tolist(), state["narrative"]
            )
            print(f"[Eval] Persona feedback: {feedback}")
            return {"feedback": feedback}

        def apply_feedback(state: EvalState):
            if state["done"] or state.get("error"):
                return {}

            if not state.get("feedback"):
                print("[Eval] ERROR: No feedback available to apply")
                return {"done": True, "error": "missing_feedback"}

            self.orchestrator.apply_feedback(state["feedback"])
            diffs = np.abs(self.orchestrator.cpm.vector - target_vector)
            print(
                f"[Eval] After update | CPM: {self.orchestrator.cpm.get_state()} | diffs: {np.round(diffs, 3).tolist()}"
            )
            return {"step": state["step"] + 1}

        def check_satisfied(state: EvalState):
            if state.get("error"):
                print(f"[Eval] ERROR: Stopping due to error - {state['error']}")
                return {"done": True, "should_finalize": False}

            if state["done"]:
                return {"done": True, "should_finalize": False}

            if self._is_satisfied(target_vector):
                print("[Eval] Satisfied (within tolerance).")

                # Once aligned, generate one fresh "final" narrative at the final CPM state.
                return {"done": True, "should_finalize": True}

            if state["step"] >= state["max_steps"]:
                print("[Eval] Stopping (max_steps reached).")
                return {"done": True, "should_finalize": False}

            return {"done": False, "should_finalize": False}

        def final_generate(state: EvalState):
            if state.get("error"):
                print(f"[Eval] ERROR: Cannot finalize due to error - {state['error']}")
                return {}

            target_style = self.orchestrator.cpm.get_state()
            results = self.orchestrator._generate_verified(ground_truth, target_style)
            if results["status"] != "success":
                print(
                    f"[Eval] ERROR: Final generation failed - {results.get('narrative', 'unknown error')}"
                )
                return {
                    "done": True,
                    "should_finalize": False,
                    "error": "final_generation_failed",
                }

            final_text = results["narrative"]
            print("\n" + "*" * 30 + " FINAL VERIFIED NARRATIVE " + "*" * 30)
            print(final_text)
            print("*" * 86)
            return {"final_narrative": final_text, "should_finalize": False}

        graph = StateGraph(EvalState)
        graph.add_node("generate_narrative", generate_narrative)
        graph.add_node("persona_feedback", persona_feedback)
        graph.add_node("apply_feedback", apply_feedback)
        graph.add_node("check_satisfied", check_satisfied)
        graph.add_node("final_generate", final_generate)

        graph.set_entry_point("generate_narrative")
        graph.add_edge("generate_narrative", "persona_feedback")
        graph.add_edge("persona_feedback", "apply_feedback")
        graph.add_edge("apply_feedback", "check_satisfied")
        graph.add_conditional_edges(
            "check_satisfied",
            lambda state: (
                "finalize"
                if state.get("should_finalize", False)
                else ("done" if state["done"] else "continue")
            ),
            {
                "finalize": "final_generate",
                "done": END,
                "continue": "generate_narrative",
            },
        )
        graph.add_edge("final_generate", END)
        return graph.compile()

    def run(self, persona_name: str, instance_idx: int, max_steps: int):
        personas = self.cfg.get("personas", {})
        if persona_name not in personas:
            raise ValueError(f"Persona '{persona_name}' not found in config.")

        target_vector = np.array(personas[persona_name], dtype=float)
        print(
            f"[Eval] Persona='{persona_name}' target={target_vector.tolist()} tol={self.alignment_tol} max_steps={max_steps}"
        )

        self.orchestrator.reset_style()
        ground_truth = self.orchestrator.prepare_case(instance_idx)

        if self._is_satisfied(target_vector):
            print("[Eval] Already satisfied at initial CPM state.")

            target_style = self.orchestrator.cpm.get_state()
            results = self.orchestrator._generate_verified(ground_truth, target_style)
            if results.get("status") == "success":
                print("\n" + "*" * 30 + " FINAL VERIFIED NARRATIVE " + "*" * 30)
                print(results["narrative"])
                print("*" * 86)
                return {
                    "steps": 0,
                    "success": True,
                    "final_cpm_vector": self.orchestrator.cpm.vector.tolist(),
                    "error": "",
                }
            else:
                print(
                    f"[Eval] ERROR: Generation failed even though alignment satisfied - {results.get('narrative', 'unknown error')}"
                )
                print("[Eval] Cannot proceed with evaluation.")
                return {
                    "steps": 0,
                    "success": False,
                    "final_cpm_vector": self.orchestrator.cpm.vector.tolist(),
                    "error": "generation_failed",
                }

        graph = self._build_graph(persona_name, target_vector, ground_truth)
        initial_state: EvalState = {
            "step": 0,
            "max_steps": max_steps,
            "done": False,
            "should_finalize": False,
            "narrative": "",
            "final_narrative": "",
            "feedback": "",
            "error": "",
        }
        result = graph.invoke(initial_state)

        if result.get("error"):
            print(f"[Eval] ERROR: Workflow failed with error: {result['error']}")
            return {
                "steps": int(result.get("step", 0)),
                "success": False,
                "final_cpm_vector": self.orchestrator.cpm.vector.tolist(),
                "error": result.get("error", "unknown_error"),
            }

        success = bool(result.get("final_narrative"))
        return {
            "steps": int(result.get("step", 0)),
            "success": success,
            "final_cpm_vector": self.orchestrator.cpm.vector.tolist(),
            "error": "",
        }


def _pick_top_k_idxs(
    orchestrator: HumanInteractiveOrchestrator, top_k: int
) -> list[int]:
    with torch.no_grad():
        X_test_t = torch.tensor(orchestrator.backend.X_test.values, dtype=torch.float32)
        probs = orchestrator.backend.model(X_test_t).numpy().flatten()
    ranked = np.argsort(-probs)
    k = max(1, min(int(top_k), len(ranked)))
    return [int(idx) for idx in ranked[:k]]


def _pick_best_idx(orchestrator: HumanInteractiveOrchestrator) -> int:
    return _pick_top_k_idxs(orchestrator, top_k=1)[0]


def main():
    parser = argparse.ArgumentParser(description="Agent-as-a-Judge evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["diabetes", "lendingclub"],
        required=True,
        help="Dataset to use: 'diabetes' or 'lendingclub'.",
    )
    parser.add_argument(
        "--persona",
        type=str,
        required=False,
        help="Persona name from the selected dataset config.",
    )
    parser.add_argument(
        "--instance_idx",
        type=int,
        default=None,
        help="Index of instance to explain. Default is highest-risk case.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Max feedback steps before stopping (overrides config).",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch evaluation on top-K highest-risk samples.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Number of highest-risk samples to evaluate (batch mode).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV path for batch results (defaults by dataset).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume batch mode by skipping samples already in output CSV.",
    )
    args = parser.parse_args()

    if args.dataset == "diabetes":
        config_path = "src/config/config_diabetes.yaml"
    elif args.dataset == "lendingclub":
        config_path = "src/config/config_lendingclub.yaml"
    else:
        config_path = "src/config/config.yaml"

    if args.output_csv is None:
        args.output_csv = f"data/results/agent_judge_eval_{args.dataset}.csv"

    evaluator = AgentJudgeEvaluator(config_path)
    available_personas = list(evaluator.cfg.get("personas", {}).keys())
    max_steps = (
        args.max_steps
        if args.max_steps is not None
        else evaluator.cfg.get("orchestrator", {}).get("max_steps", 20)
    )

    if args.batch:
        personas = evaluator.cfg.get("personas", {})
        persona_names = list(personas.keys())
        if not persona_names:
            raise ValueError("No personas found in config.")

        rng = np.random.default_rng(evaluator.cfg.get("seed", 42))
        top_k_idxs = _pick_top_k_idxs(evaluator.orchestrator, args.top_k)

        output_dir = os.path.dirname(args.output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        existing_samples: set[int] = set()
        if args.resume and os.path.exists(args.output_csv):
            with open(args.output_csv, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        existing_samples.add(int(row.get("sample_idx", "")))
                    except (TypeError, ValueError):
                        continue

        remaining_idxs = [idx for idx in top_k_idxs if idx not in existing_samples]
        if args.resume and existing_samples:
            print(
                f"[Batch] Resuming: {len(existing_samples)} completed, "
                f"{len(remaining_idxs)} remaining (top_k={args.top_k})"
            )

        write_header = not os.path.exists(args.output_csv) or not args.resume
        mode = "a" if args.resume else "w"
        with open(args.output_csv, mode, newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sample_idx",
                    "persona",
                    "steps",
                    "final_cpm_vector",
                    "success",
                ],
            )
            if write_header:
                writer.writeheader()

            for sample_idx in remaining_idxs:
                persona = rng.choice(persona_names)
                print(
                    f"\n[Batch] sample_idx={sample_idx} persona={persona} (top_k={args.top_k})"
                )
                result = evaluator.run(persona, sample_idx, max_steps)
                writer.writerow(
                    {
                        "sample_idx": sample_idx,
                        "persona": persona,
                        "steps": result["steps"],
                        "final_cpm_vector": json.dumps(result["final_cpm_vector"]),
                        "success": result["success"],
                    }
                )
                f.flush()
    else:
        if not args.persona:
            raise ValueError(
                "--persona is required unless --batch is set. "
                f"Available personas: {', '.join(available_personas)}"
            )
        if available_personas and args.persona not in available_personas:
            raise ValueError(
                f"Invalid persona '{args.persona}'. "
                f"Available personas: {', '.join(available_personas)}"
            )

        instance_idx = (
            args.instance_idx
            if args.instance_idx is not None
            else _pick_best_idx(evaluator.orchestrator)
        )
        result = evaluator.run(args.persona, instance_idx, max_steps)
        print(f"[Evaluation] Steps taken: {result['steps']}")


if __name__ == "__main__":
    main()
