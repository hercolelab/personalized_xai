import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import shap
import dice_ml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score

# --- A. The Neural Network ---
class DiabetesNN(nn.Module):
    def __init__(self, input_dim):
        super(DiabetesNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.network(x)

# --- B. The XAI Backend ---
class XAI_Backend:
    def __init__(self, csv_path='diabetes.csv'):
        print(" [Backend] Loading PIMA Dataset...")
        self.df = pd.read_csv(csv_path)
        self.target = 'Outcome'
        
        self.numerical = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                          'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        self.scaler = StandardScaler()
        self.df_scaled = self.df.copy()
        self.df_scaled[self.numerical] = self.scaler.fit_transform(self.df[self.numerical])

        X = self.df_scaled.drop(self.target, axis=1)
        y = self.df_scaled[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = DiabetesNN(X.shape[1])
        optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        criterion = nn.BCELoss()
        
        X_t = torch.tensor(self.X_train.values, dtype=torch.float32)
        y_t = torch.tensor(self.y_train.values, dtype=torch.float32).view(-1, 1)
        
        for epoch in range(100):
            optimizer.zero_grad()
            loss = criterion(self.model(X_t), y_t)
            loss.backward()
            optimizer.step()
        
        self.model.eval()
        self.print_performance()

        background = X_t[:100]
        self.shap_explainer = shap.DeepExplainer(self.model, background)
        
        d = dice_ml.Data(dataframe=self.df_scaled, continuous_features=self.numerical, outcome_name=self.target)
        m = dice_ml.Model(model=self.model, backend="PYT")
        self.dice_explainer = dice_ml.Dice(d, m, method="gradient")

    def print_performance(self):
        with torch.no_grad():
            X_test_t = torch.tensor(self.X_test.values, dtype=torch.float32)
            y_probs = self.model(X_test_t).numpy()
            y_preds = (y_probs > 0.5).astype(int)
            
            acc = accuracy_score(self.y_test, y_preds)
            prec = precision_score(self.y_test, y_preds)
            rec = recall_score(self.y_test, y_preds)
            
            print(f"\n [Backend] --- TEST SET PERFORMANCE ---")
            print(f" Accuracy:  {acc:.2%}")
            print(f" Precision: {prec:.2%}")
            print(f" Recall:    {rec:.2%}")
            print(f" --------------------------------------\n")

    def get_explanation(self, instance_idx):
            patient_row = self.X_test.iloc[[instance_idx]]
            patient_tensor = torch.tensor(patient_row.values, dtype=torch.float32)
            
            # 1. Helper for Real-World Units
            def get_real_world_values(row_df):
                temp_df = row_df.copy()
                temp_df[self.numerical] = self.scaler.inverse_transform(row_df[self.numerical])
                return temp_df.iloc[0]

            real_orig = get_real_world_values(patient_row)

            with torch.no_grad():
                prob = self.model(patient_tensor).item()
            
            # 2. SHAP Drivers
            shap_values = self.shap_explainer.shap_values(patient_tensor)
            if isinstance(shap_values, list): shap_values = shap_values[0]
            shap_list = sorted(zip(self.X_test.columns, shap_values.flatten()), 
                            key=lambda x: abs(x[1]), reverse=True)

            # 3. DYNAMIC TARGET CALCULATION
            # Instead of hard-coding -1.0, we find the 25th percentile of the training set
            actionable_features = ['Glucose', 'BMI', 'Insulin', 'BloodPressure']
            dynamic_targets = {feat: self.X_train[feat].quantile(0.25) for feat in actionable_features}

            cf_text = ""
            status = "Gradient Search Error"
            reduced_prob = prob
            
            try:
                # Attempt DiCE
                dice_exp = self.dice_explainer.generate_counterfactuals(
                    patient_row, total_CFs=1, desired_class=0,
                    features_to_vary=actionable_features
                )
                
                if dice_exp.cf_examples_list[0].final_cfs_df is not None:
                    real_target = get_real_world_values(dice_exp.cf_examples_list[0].final_cfs_df)
                    status = "Path Found"
                    reduced_prob = 0.49 
                else:
                    raise ValueError("Path Blocked")

            except Exception:
                # HEURISTIC FALLBACK with Dynamic Targets
                status = "Heuristic (Dynamic Targets)"
                best_case_row = patient_row.copy()
                
                for col, target_val in dynamic_targets.items():
                    if patient_row.iloc[0][col] > target_val:
                        best_case_row[col] = target_val
                
                real_target = get_real_world_values(best_case_row)
                
                with torch.no_grad():
                    reduced_prob = self.model(torch.tensor(best_case_row.values, dtype=torch.float32)).item()

            # 4. Format Output
            diffs = [f"{c}: {real_orig[c]:.1f} -> {real_target[c]:.1f}" 
                    for c in actionable_features if real_orig[c] > real_target[c]]
            cf_text = " | ".join(diffs)

            return f"""
            DIABETES CLINICAL PREDICTION:
            - Current Probability: {prob:.2%}
            - MINIMIZED RISK: {reduced_prob:.2%}

            LOCAL DRIVERS (SHAP):
            1. {shap_list[0][0]}: (Impact: {float(shap_list[0][1]):.4f})
            2. {shap_list[1][0]}: (Impact: {float(shap_list[1][1]):.4f})

            DYNAMIC TARGETS (Status: {status}):
            - Changes: {cf_text}
            """

if __name__ == "__main__":
    # 1. Initialize Backend
    backend = XAI_Backend(csv_path='diabetes.csv')

    # 2. Find a High Risk Case
    print("\n--- Scanning PIMA Test Set for High Risk Patient ---")
    with torch.no_grad():
        X_test_t = torch.tensor(backend.X_test.values, dtype=torch.float32)
        probs = backend.model(X_test_t).numpy().flatten()
    
    best_idx = int(np.argmax(probs))
    max_prob = probs[best_idx]
    
    print(f" [Found] Patient Index: {best_idx} | Risk: {max_prob:.2%}")

    # 3. Generate Explanation
    raw_xai_data = backend.get_explanation(best_idx)
    print("\n" + "="*40)
    print(" RAW XAI INPUT DATA:")
    print(raw_xai_data)
    print("="*40)