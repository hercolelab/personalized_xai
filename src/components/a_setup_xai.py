import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import shap
import dice_ml
import yaml
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
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.df = pd.read_csv(self.cfg["data"]["csv_path"])
        self.target_col = self.cfg["data"]["target"]

        self._feature_discrimination()
        self._prepare_data()
        self._build_and_train_model()
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

    def _prepare_data(self):
        self.scaler = StandardScaler()
        self.df_scaled = self.df.copy()
        if self.numerical:
            self.df_scaled[self.numerical] = self.scaler.fit_transform(
                self.df[self.numerical]
            )
        for col in self.categorical:
            le = LabelEncoder()
            self.df_scaled[col] = le.fit_transform(self.df[col])

        X = self.df_scaled.drop(self.target_col, axis=1)
        y = self.df_scaled[self.target_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.cfg["data"]["test_size"], random_state=self.cfg["seed"]
        )

    def _build_and_train_model(self):
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
        try:
            dice_exp = self.dice_explainer.generate_counterfactuals(
                patient_row,
                total_CFs=1,
                desired_class=0,
                features_to_vary=dynamic_actionable,
            )
            if dice_exp.cf_examples_list[0].final_cfs_df is not None:
                cf_df = dice_exp.cf_examples_list[0].final_cfs_df
                real_target = get_real_world_values(cf_df)
                status = "Path Found"
                with torch.no_grad():
                    reduced_prob = self.model(
                        torch.tensor(
                            cf_df.drop(self.target_col, axis=1).values,
                            dtype=torch.float32,
                        )
                    ).item()
        except Exception:
            status = "Heuristic"
            best_case = patient_row.copy()
            for f in dynamic_actionable:
                if f in self.numerical:
                    val = self.X_train[f].quantile(0.25)
                    if patient_row.iloc[0][f] > val:
                        best_case[f] = val
            real_target = get_real_world_values(best_case)
            with torch.no_grad():
                reduced_prob = self.model(
                    torch.tensor(best_case.values, dtype=torch.float32)
                ).item()

        # --- SELECT ONLY CHANGED FEATURES ---
        # We only want features where the difference is significant (> 0.1 units)
        changed_features = [
            f for f in dynamic_actionable if abs(real_orig[f] - real_target[f]) > 0.1
        ]

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
                "minimized": round(reduced_prob * 100, 2),
            },
        }

        diff_str = " | ".join(
            [f"{f}: {real_orig[f]:.1f}->{real_target[f]:.1f}" for f in changed_features]
        )
        raw_text = f"RESULTS:\n- Prob: {prob:.2%} -> {reduced_prob:.2%}\n- Changes: {diff_str if diff_str else 'None'}"

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
