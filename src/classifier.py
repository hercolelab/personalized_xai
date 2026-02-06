"""
Script temporaneo per esportare le probabilità di rischio per tutte le istanze del dataset.
Uso: python src/export_risk_probabilities.py --dataset diabetes
     python src/export_risk_probabilities.py --dataset lendingclub
"""
import argparse
import pandas as pd
import numpy as np
import torch
from src.components.a_setup_xai import XAI_Backend


def main():
    parser = argparse.ArgumentParser(description="Export risk probabilities to CSV")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["diabetes", "lendingclub"],
        required=True,
        help="Dataset to use: 'diabetes' or 'lendingclub'.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path. Default: data/raw/{dataset}_risk_probabilities.csv",
    )
    args = parser.parse_args()

    # Load config based on dataset
    if args.dataset == "diabetes":
        config_file = "src/config/config_diabetes.yaml"
    elif args.dataset == "lendingclub":
        config_file = "src/config/config_lendingclub.yaml"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"[Loading] Initializing backend for {args.dataset}...")
    print(f"[Training] Training model and saving to src/components/clf/...")
    backend = XAI_Backend(config_path=config_file, force_retrain=True)

    # Calculate probabilities for all test instances
    print(f"[Computing] Calculating risk probabilities for {len(backend.X_test)} instances...")
    X_test_t = torch.tensor(backend.X_test.values, dtype=torch.float32)
    
    with torch.no_grad():
        probs = backend.model(X_test_t).numpy().flatten()

    # Create output DataFrame
    # Reset index to get sequential positional indices (0, 1, 2, ...)
    # This matches the instance_idx used in orchestrator.py with --instance_idx
    output_df = backend.X_test.reset_index(drop=True).copy()
    
    # Add the positional index as first column (this is what you use with --instance_idx)
    output_df.insert(0, 'instance_idx', range(len(output_df)))
    
    output_df['risk_probability'] = probs
    
    # Also include the actual target values if available
    if hasattr(backend, 'y_test'):
        output_df['actual_target'] = backend.y_test.values

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = f"data/raw/{args.dataset}_risk_probabilities.csv"

    # Save to CSV (don't save index since we have instance_idx column)
    output_df.to_csv(output_path, index=False)
    print(f"[Saved] Risk probabilities exported to: {output_path}")
    print(f"[Stats] Min probability: {probs.min():.4f}, Max: {probs.max():.4f}, Mean: {probs.mean():.4f}")
    print(f"[Stats] Total instances: {len(probs)}")


if __name__ == "__main__":
    main()
