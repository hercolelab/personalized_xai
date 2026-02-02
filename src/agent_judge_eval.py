import argparse
import json
from typing import TypedDict

import numpy as np
import torch
import yaml
from langgraph.graph import END, StateGraph

from src.components.ollama_client import OllamaClient
from src.orchestrator import HumanInteractiveOrchestrator


class PersonaFeedbackAgent:
    def __init__(
        self,
        config_path="src/config/config.yaml",
        prompt_path="src/prompts/prompts.yaml",
    ):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        with open(prompt_path, "r") as f:
            prompts = yaml.safe_load(f)
            self.prompt_template = prompts["system_prompts"]["persona_feedback"][
                "prompt"
            ]

        self.model_name = self.cfg["verifier"].get("llm_judge_model", "llama3")
        self.ollama = OllamaClient.from_model(self.model_name)

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
        prompt_path="src/prompts/prompts.yaml",
    ):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

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
                return 0
            else:
                print(
                    f"[Eval] ERROR: Generation failed even though alignment satisfied - {results.get('narrative', 'unknown error')}"
                )
                print("[Eval] Cannot proceed with evaluation.")
                return -1

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
            return -1

        return int(result.get("step", 0))


def _pick_best_idx(orchestrator: HumanInteractiveOrchestrator) -> int:
    with torch.no_grad():
        X_test_t = torch.tensor(orchestrator.backend.X_test.values, dtype=torch.float32)
        probs = orchestrator.backend.model(X_test_t).numpy().flatten()
    return int(np.argmax(probs))


def main():
    parser = argparse.ArgumentParser(description="Agent-as-a-Judge evaluation")
    parser.add_argument(
        "--persona",
        type=str,
        required=True,
        help="Persona name from config.yaml (e.g., patient, clinician).",
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
    args = parser.parse_args()

    config_path = "src/config/config.yaml"
    evaluator = AgentJudgeEvaluator(config_path)
    max_steps = (
        args.max_steps
        if args.max_steps is not None
        else evaluator.cfg.get("orchestrator", {}).get("max_steps", 20)
    )

    instance_idx = (
        args.instance_idx
        if args.instance_idx is not None
        else _pick_best_idx(evaluator.orchestrator)
    )

    steps_taken = evaluator.run(args.persona, instance_idx, max_steps)
    print(f"[Evaluation] Steps taken: {steps_taken}")


if __name__ == "__main__":
    main()
