import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import shap
import dice_ml
import yaml
import os
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super(MLP, self).__init__()
        layers = []
        last_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            last_dim = h_dim
        layers.append(nn.Linear(last_dim, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        if x.ndimension() == 1:
            x = x.unsqueeze(0)
        return self.network(x)


class XAI_Backend:
    def __init__(self, config_path="config.yaml", force_retrain=False):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.df = pd.read_csv(self.cfg["data"]["csv_path"])
        self.target_col = self.cfg["data"]["target"]

        self._feature_discrimination()
        # Try to load model first (includes scaler and label encoders)
        model_loaded = False
        if not force_retrain:
            model_path = self._get_model_path()
            model_loaded = self._load_model(model_path)
        
        # Prepare data (use loaded scaler/encoders if model was loaded)
        self._prepare_data(use_loaded_preprocessors=model_loaded)
        
        # Build/train model if not loaded
        if not model_loaded:
            self._build_and_train_model(force_retrain=force_retrain)
        
        self._init_explainers()

    def _feature_discrimination(self):
        X_raw = self.df.drop(self.target_col, axis=1)
        self.numerical = []
        self.categorical = []
        for col in X_raw.columns:
            if X_raw[col].dtype in ["float64", "float32"] or X_raw[col].nunique() >= 15:
                self.numerical.append(col)
            else:
                self.categorical.append(col)

    def _prepare_data(self, use_loaded_preprocessors=False):
        """Prepare data, using loaded scaler/label encoders if available."""
        if not use_loaded_preprocessors:
            # Fit new scaler and label encoders
            self.scaler = StandardScaler()
            self.label_encoders = {}
        # else: scaler and label_encoders were loaded in _load_model
        
        self.df_scaled = self.df.copy()
        if self.numerical:
            if use_loaded_preprocessors:
                self.df_scaled[self.numerical] = self.scaler.transform(
                    self.df[self.numerical]
                )
            else:
                self.df_scaled[self.numerical] = self.scaler.fit_transform(
                    self.df[self.numerical]
                )
        
        for col in self.categorical:
            if use_loaded_preprocessors:
                le = self.label_encoders[col]
                self.df_scaled[col] = le.transform(self.df[col])
            else:
                le = LabelEncoder()
                self.df_scaled[col] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le

        X = self.df_scaled.drop(self.target_col, axis=1)
        y = self.df_scaled[self.target_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.cfg["data"]["test_size"], random_state=self.cfg["seed"]
        )

    def _get_model_path(self):
        """Get the path where the model should be saved/loaded."""
        # Infer dataset name from config path or csv path
        csv_path = self.cfg.get("data", {}).get("csv_path", "")
        if "diabetes" in csv_path.lower():
            dataset_name = "diabetes"
        elif "lendingclub" in csv_path.lower():
            dataset_name = "lendingclub"
        else:
            # Fallback: use config filename
            dataset_name = Path(self.cfg.get("data", {}).get("csv_path", "unknown")).stem.replace("_cleaned", "")
        
        model_dir = Path(__file__).parent / "clf"
        model_dir.mkdir(exist_ok=True)
        return model_dir / f"{dataset_name}_model.pth"

    def _save_model(self, model_path):
        """Save model, scaler, label encoders, and metadata."""
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "model_config": {
                "input_dim": self.X_train.shape[1],
                "hidden_layers": self.cfg["model"]["hidden_layers"],
            },
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
            "numerical": self.numerical,
            "categorical": self.categorical,
            "target_col": self.target_col,
        }
        torch.save(save_dict, model_path)
        print(f"[Model] Saved model to {model_path}")

    def _load_model(self, model_path):
        """Load model, scaler, label encoders, and metadata. Returns True if successful."""
        if not model_path.exists():
            return False
        
        print(f"[Model] Loading model from {model_path}")
        # weights_only=False needed because we save sklearn objects (StandardScaler, LabelEncoder)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        
        # Load scaler and label encoders first (needed for _prepare_data)
        self.scaler = checkpoint["scaler"]
        self.label_encoders = checkpoint["label_encoders"]
        self.numerical = checkpoint["numerical"]
        self.categorical = checkpoint["categorical"]
        self.target_col = checkpoint["target_col"]
        
        # Reconstruct model (will be set in _build_and_train_model if needed)
        model_config = checkpoint["model_config"]
        self.model = MLP(model_config["input_dim"], model_config["hidden_layers"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        print(f"[Model] Model loaded successfully")
        return True

    def _build_and_train_model(self, force_retrain=False):
        """Build and train model. Model should already be loaded if available."""
        # If model was already loaded in __init__, skip training
        if hasattr(self, 'model') and self.model is not None and not force_retrain:
            return
        
        # Train new model
        print(f"[Model] Training new model...")
        self.model = MLP(self.X_train.shape[1], self.cfg["model"]["hidden_layers"])
        optimizer = optim.Adam(self.model.parameters(), lr=self.cfg["model"]["lr"])
        criterion = nn.BCELoss()
        X_t = torch.tensor(self.X_train.values, dtype=torch.float32)
        y_t = torch.tensor(self.y_train.values, dtype=torch.float32).view(-1, 1)
        for _ in range(self.cfg["model"]["epochs"]):
            optimizer.zero_grad()
            criterion(self.model(X_t), y_t).backward()
            optimizer.step()
        self.model.eval()
        
        # Save the trained model
        model_path = self._get_model_path()
        self._save_model(model_path)

    def _init_explainers(self):
        background = torch.tensor(
            self.X_train.values[: self.cfg["xai"]["shap_background_samples"]],
            dtype=torch.float32,
        )
        self.shap_explainer = shap.DeepExplainer(self.model, background)
        d = dice_ml.Data(
            dataframe=self.df_scaled,
            continuous_features=self.numerical,
            outcome_name=self.target_col,
        )
        m = dice_ml.Model(model=self.model, backend="PYT")
        self.dice_explainer = dice_ml.Dice(d, m, method=self.cfg["xai"]["dice_method"])

    def get_explanation(self, instance_idx):
        patient_row = self.X_test.iloc[[instance_idx]]
        patient_tensor = torch.tensor(patient_row.values, dtype=torch.float32)

        def get_real_world_values(row_df):
            temp_df = row_df.copy()
            if self.numerical:
                temp_df[self.numerical] = self.scaler.inverse_transform(
                    row_df[self.numerical]
                )
            return temp_df.iloc[0]

        real_orig = get_real_world_values(patient_row)
        with torch.no_grad():
            prob = self.model(patient_tensor).item()
        
        # Calculate binary class with threshold 0.5
        threshold = 0.5
        original_class = 1 if prob > threshold else 0
        desired_class = 1 - original_class  # Opposite class

        # SHAP
        shap_values = self.shap_explainer.shap_values(patient_tensor)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_list = sorted(
            zip(self.X_test.columns, shap_values.flatten()),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        # Actionable Binding
        actionable_pool = self.cfg["data"]["actionable_features"]
        dynamic_actionable = [
            f
            for f, imp in shap_list
            if f in actionable_pool
            and abs(imp) > self.cfg["xai"]["shap_relevance_threshold"]
        ]
        if not dynamic_actionable:
            dynamic_actionable = actionable_pool

        # Counterfactual Logic
        status, reduced_prob, real_target = "Error", prob, real_orig.copy()
        cf_df_scaled = None  # Keep reference to scaled counterfactual for BMI correction
        counterfactual_class = None  # Will be calculated after getting reduced_prob
        try:
            dice_exp = self.dice_explainer.generate_counterfactuals(
                patient_row,
                total_CFs=1,
                desired_class=desired_class,  # Opposite class to original
                features_to_vary=dynamic_actionable,
            )
            if dice_exp.cf_examples_list[0].final_cfs_df is not None:
                cf_df = dice_exp.cf_examples_list[0].final_cfs_df
                cf_df_scaled = cf_df.drop(self.target_col, axis=1) if self.target_col in cf_df.columns else cf_df
                real_target = get_real_world_values(cf_df)
                status = "Path Found"
                with torch.no_grad():
                    reduced_prob = self.model(
                        torch.tensor(cf_df_scaled.values, dtype=torch.float32)
                    ).item()
                
                # Verify that the counterfactual actually changed the class
                counterfactual_class = 1 if reduced_prob > threshold else 0
                if counterfactual_class != desired_class:
                    # DiCE found a counterfactual but it didn't cross the threshold
                    # Fall back to heuristic mode
                    print(f"[Info] DiCE generated counterfactual with prob={reduced_prob:.3f} (class={counterfactual_class}), "
                          f"but desired class was {desired_class}. Falling back to heuristic mode.")
                    raise ValueError("Counterfactual did not flip the class")
            else:
                # DiCE didn't find any counterfactual
                raise ValueError("DiCE could not generate counterfactual")
        except Exception as e:
            # Use heuristic mode when DiCE fails or doesn't flip the class
            status = "Heuristic"
            print(f"[Heuristic] Using heuristic counterfactual generation (reason: {type(e).__name__})")
            best_case = patient_row.copy()
            # Heuristic: if desired_class is 0 (low risk), reduce values; if 1 (high risk), increase values
            for f in dynamic_actionable:
                if f in self.numerical:
                    if desired_class == 0:
                        # Reduce risk: use lower percentile (25th)
                        val = self.X_train[f].quantile(0.25)
                        if patient_row.iloc[0][f] > val:
                            best_case[f] = val
                    else:
                        # Increase risk: use higher percentile (75th)
                        val = self.X_train[f].quantile(0.75)
                        if patient_row.iloc[0][f] < val:
                            best_case[f] = val
            cf_df_scaled = best_case  # Keep reference for BMI correction
            real_target = get_real_world_values(best_case)
            with torch.no_grad():
                reduced_prob = self.model(
                    torch.tensor(best_case.values, dtype=torch.float32)
                ).item()
            # Reset counterfactual_class since we're using heuristic
            counterfactual_class = None

        # --- PREVENT BMI INCREASE WHEN REDUCING RISK ---
        # BMI should never increase when trying to reduce diabetes risk (desired_class == 0)
        # Note: This is domain-specific logic for diabetes dataset
        if "BMI" in real_target and "BMI" in real_orig and cf_df_scaled is not None and desired_class == 0:
            if real_target["BMI"] > real_orig["BMI"]:
                # Keep BMI at original value (don't increase it when reducing risk)
                real_target["BMI"] = real_orig["BMI"]
                # Correct BMI in scaled space and recalculate probability
                if "BMI" in cf_df_scaled.columns:
                    cf_df_scaled = cf_df_scaled.copy()
                    cf_df_scaled["BMI"] = patient_row.iloc[0]["BMI"]  # Use original BMI value
                    with torch.no_grad():
                        reduced_prob = self.model(
                            torch.tensor(cf_df_scaled.values, dtype=torch.float32)
                        ).item()

        # --- SELECT ONLY CHANGED FEATURES ---
        # We only want features where the difference is significant (> 0.1 units)
        changed_features = [
            f for f in dynamic_actionable if abs(real_orig[f] - real_target[f]) > 0.1
        ]

        # Calculate counterfactual class (if not already calculated in try block)
        if counterfactual_class is None:
            counterfactual_class = 1 if reduced_prob > threshold else 0
        
        # Verify class flip occurred
        class_flipped = (counterfactual_class != original_class)
        
        ground_truth = {
            "current": {k: round(float(real_orig[k]), 1) for k in self.numerical},
            "target": {
                k: round(float(real_target[k]), 1) for k in changed_features
            },  # ONLY CHANGED
            "shap_impacts": {
                feat: round(float(imp), 4) for feat, imp in shap_list
            },  # All features ordered by magnitude
            "probabilities": {
                "current": round(prob * 100, 2),
                "counterfactual": round(reduced_prob * 100, 2),
            },
            "classes": {
                "current": int(original_class),
                "counterfactual": int(counterfactual_class),
                "desired": int(desired_class),
                "flipped": bool(class_flipped),  # True if class actually changed
            },
        }

        diff_str = " | ".join(
            [f"{f}: {real_orig[f]:.1f}->{real_target[f]:.1f}" for f in changed_features]
        )
        flip_status = "✓ FLIPPED" if class_flipped else "✗ NOT FLIPPED"
        raw_text = f"RESULTS:\n- Prob: {prob:.2%} (Class {original_class}) -> {reduced_prob:.2%} (Class {counterfactual_class})\n- Desired Class: {desired_class} | {flip_status}\n- Changes: {diff_str if diff_str else 'None'}"

        return raw_text, ground_truth


if __name__ == "__main__":
    backend = XAI_Backend(config_path="src/config/config.yaml")
    # Find highest risk patient to test
    X_test_t = torch.tensor(backend.X_test.values, dtype=torch.float32)
    with torch.no_grad():
        probs = backend.model(X_test_t).numpy().flatten()

    idx = int(np.argmax(probs))
    print(f"\n [Test] Explaining Patient {idx} (Risk: {probs[idx]:.2%})")
    print(backend.get_explanation(idx))
