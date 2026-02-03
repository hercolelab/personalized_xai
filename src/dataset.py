import argparse
import pandas as pd
import os


def clean_dataset(dataset_name):
    """
    Load and clean a dataset based on the dataset name.

    Args:
        dataset_name (str): Name of the dataset to load and clean
    """
    # Get the main directory (parent of src)
    main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if dataset_name == "diabetes":
        # Load diabetes.csv from main directory
        input_path = os.path.join(main_dir, "data/raw/diabetes.csv")
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Dataset file not found: {input_path}")

        # Load the dataset
        df = pd.read_csv(input_path)

        # Remove entries where Glucose, BloodPressure, SkinThickness, BMI, or Insulin have value 0
        columns_to_check = [
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "BMI",
            "Insulin",
        ]

        # Filter out rows where any of these columns have value 0
        df_cleaned = df[~((df[columns_to_check] == 0).any(axis=1))]

        # Save the cleaned dataset
        output_path = os.path.join(main_dir, "data/raw/diabetes_cleaned.csv")
        df_cleaned.to_csv(output_path, index=False)

        print(f"Cleaned dataset saved to: {output_path}")
        print(
            f"Original rows: {len(df)}, Cleaned rows: {len(df_cleaned)}, Removed: {len(df) - len(df_cleaned)}"
        )

    else:
        raise ValueError(
            f"Dataset '{dataset_name}' is not supported. Currently only 'diabetes' is supported."
        )


def main():
    parser = argparse.ArgumentParser(description="Load and clean datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset to load and clean",
    )

    args = parser.parse_args()

    clean_dataset(args.dataset)


if __name__ == "__main__":
    main()
