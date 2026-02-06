import argparse
import json
import os
import random
import numpy as np
import torch
import yaml
from pathlib import Path
from termcolor import colored
from src.orchestrator import XAI_Orchestrator


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Baseline Experiment for XAI Pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["diabetes", "lendingclub"],
        required=True,
        help="Dataset to use: 'diabetes' or 'lendingclub'.",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=100,
        help="Number of iterations to run the baseline experiment (default: 100).",
    )
    parser.add_argument(
        "--instance_idx",
        type=int,
        default=None,
        help="Index of instance to explain. If not provided, uses highest-risk case.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Load config based on dataset
    if args.dataset == "diabetes":
        config_file = "src/config/config_diabetes.yaml"
    elif args.dataset == "lendingclub":
        config_file = "src/config/config_lendingclub.yaml"
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Load config to get personas
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    # Get available personas from config
    available_personas = list(cfg.get("personas", {}).keys())
    if not available_personas:
        raise ValueError(f"No personas defined in config file: {config_file}")

    print(
        colored(
            f"[Baseline Experiment] Dataset: {args.dataset} | Instances: {args.num_iterations}",
            "cyan",
            attrs=["bold"],
        )
    )
    print(colored(f"[Baseline Experiment] Available personas: {available_personas}", "cyan"))

    # Initialize orchestrator in baseline mode
    orchestrator = XAI_Orchestrator(config_file, baseline=True)

    # Determine instances to use
    test_size = len(orchestrator.backend.X_test)
    
    if args.instance_idx is not None:
        # Use only the specified instance
        if args.instance_idx < 0 or args.instance_idx >= test_size:
            raise ValueError(
                f"instance_idx {args.instance_idx} is out of bounds. "
                f"Test set has {test_size} instances (valid indices: 0-{test_size-1})."
            )
        instance_indices = [args.instance_idx]
        print(colored(f"[Baseline Experiment] Using specified instance: {args.instance_idx}", "cyan"))
    else:
        # Randomly sample distinct instances
        num_instances = min(args.num_iterations, test_size)
        instance_indices = random.sample(range(test_size), num_instances)
        print(colored(f"[Baseline Experiment] Randomly sampled {num_instances} distinct instances", "cyan"))

    # Store results per instance
    results_per_instance = []

    # Run one generation per instance
    for idx, instance_idx in enumerate(instance_indices):
        # Randomly select a persona for this instance
        role = random.choice(available_personas)

        print(
            colored(
                f"\n{'=' * 80}\n[Instance {idx + 1}/{len(instance_indices)}] idx={instance_idx} | Role: {role}\n{'=' * 80}",
                "yellow",
                attrs=["bold"],
            )
        )

        # Get ground truth data FIRST (before running orchestrator)
        raw_xai_text, ground_truth = orchestrator.backend.get_explanation(instance_idx)
        
        # Get target style
        orchestrator.cpm.initialize(role)
        target_style = orchestrator.cpm.get_state()
        
        # Generate narrative directly (to get version WITH tags for verification)
        candidate_narrative = orchestrator.narrator.generate(ground_truth, target_style)
        
        # Print the candidate narrative
        print(colored(f"\n{'-' * 15} CANDIDATE NARRATIVE {'-' * 15}", "yellow"))
        print(candidate_narrative)
        print(colored("-" * 51, "yellow"))
        
        if not candidate_narrative:
            print(colored(f"[WARNING] Generation failed for instance {instance_idx}", "red"))
            # Store failure result
            results_per_instance.append({
                "instance_idx": instance_idx,
                "role": role,
                "target_style": target_style,
                "is_faithful": False,
                "is_complete": False,
                "is_aligned": False,
                "faithfulness_reason": "Generation failed",
                "completeness_reason": "Generation failed",
                "alignment_report": None,
            })
            continue

        # Use the narrative WITH tags for verification
        narrative = candidate_narrative

        # Verify faithfulness
        is_faithful, reason_f = orchestrator.verifier.verify_faithfulness(
            narrative, ground_truth, target_style
        )

        # Verify completeness
        is_complete, reason_c = orchestrator.verifier.verify_completeness(
            narrative, ground_truth, target_style
        )

        # Verify alignment
        is_aligned, style_report = orchestrator.verifier.verify_alignment(
            narrative, target_style
        )

        # Store results for this instance
        results_per_instance.append({
            "instance_idx": instance_idx,
            "role": role,
            "target_style": target_style,
            "is_faithful": is_faithful,
            "is_complete": is_complete,
            "is_aligned": is_aligned,
            "faithfulness_reason": reason_f if not is_faithful else None,
            "completeness_reason": reason_c if not is_complete else None,
            "alignment_report": style_report,
        })

        # Print verification results
        print(colored("\n[Verification Results]", "cyan"))
        print(colored(f"  Faithfulness: {'PASS' if is_faithful else 'FAIL'}", "green" if is_faithful else "red"))
        if not is_faithful:
            print(colored(f"    Reason: {reason_f}", "red"))
        print(colored(f"  Completeness: {'PASS' if is_complete else 'FAIL'}", "green" if is_complete else "red"))
        if not is_complete:
            print(colored(f"    Reason: {reason_c}", "red"))
        print(colored(f"  Alignment: {'PASS' if is_aligned else 'FAIL'}", "green" if is_aligned else "red"))
        if not is_aligned:
            print(colored(f"    Report: {style_report}", "red"))

    # Extract vectors from results
    instance_idx_vector = [r["instance_idx"] for r in results_per_instance]
    is_faithful_vector = [r["is_faithful"] for r in results_per_instance]
    is_complete_vector = [r["is_complete"] for r in results_per_instance]
    is_aligned_vector = [r["is_aligned"] for r in results_per_instance]
    style_vector = [
        [
            r["target_style"]["technicality"],
            r["target_style"]["verbosity"],
            r["target_style"]["depth"],
            r["target_style"]["perspective"],
        ]
        for r in results_per_instance
    ]

    # Compute aggregate statistics
    total_instances = len(results_per_instance)
    if total_instances > 0:
        faithful_count = sum(is_faithful_vector)
        complete_count = sum(is_complete_vector)
        aligned_count = sum(is_aligned_vector)
        
        faithful_rate = faithful_count / total_instances
        complete_rate = complete_count / total_instances
        aligned_rate = aligned_count / total_instances
    else:
        faithful_count = complete_count = aligned_count = 0
        faithful_rate = complete_rate = aligned_rate = 0.0

    print(
        colored(
            f"\n{'=' * 80}\n[Baseline Experiment] Results Summary\n{'=' * 80}",
            "green",
            attrs=["bold"],
        )
    )
    print(colored(f"Total Instances: {total_instances}", "cyan"))
    print(colored(f"Faithfulness Rate: {faithful_rate:.2%} ({faithful_count}/{total_instances})", "cyan"))
    print(colored(f"Completeness Rate: {complete_rate:.2%} ({complete_count}/{total_instances})", "cyan"))
    print(colored(f"Alignment Rate: {aligned_rate:.2%} ({aligned_count}/{total_instances})", "cyan"))

    # Prepare simplified results dictionary
    results_dict = {
        "dataset": args.dataset,
        "num_instances": total_instances,
        "seed": args.seed,
        "instance_idx": instance_idx_vector,
        "is_faithful": is_faithful_vector,
        "is_complete": is_complete_vector,
        "is_aligned": is_aligned_vector,
        "style": style_vector,
        "faithful_rate": faithful_rate,
        "complete_rate": complete_rate,
        "aligned_rate": aligned_rate,
    }

    # Create output directory
    output_dir = Path(f"results/baseline/{args.dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get model name from config
    model_name = cfg.get("verifier", {}).get("llm_judge_model", "unknown")
    # Clean model name for filename (replace : with -)
    model_name_clean = model_name.replace(":", "-")

    # Save results to JSON
    output_file = output_dir / f"results_{model_name_clean}_seed_{args.seed}.json"
    
    with open(output_file, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(colored(f"\n[Baseline Experiment] Results saved to: {output_file}", "green", attrs=["bold"]))


if __name__ == "__main__":
    main()
