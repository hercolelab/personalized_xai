import pandas as pd
import numpy as np
import ast

# 1. Load Data
# Assuming your file is named 'agent_eval_results.csv'
df = pd.read_csv('agent_judge_eval_lendingclub_50.csv')

df['final_cpm_vector'] = df['final_cpm_vector'].apply(lambda x: np.array(ast.literal_eval(x)))

# Define Baseline
baseline = np.array([0.0, 0.5, 0.5, 0.5])

# 2. Distance and Success Filtering
df['dist_to_baseline'] = df['final_cpm_vector'].apply(lambda v: np.linalg.norm(v - baseline))
successful_df = df[df['success'] == True].copy()

# 3. Efficiency Score (η) Calculation
# Global η = Sum of Steps / Sum of Distances
total_steps = successful_df['steps'].sum()
total_dist = successful_df['dist_to_baseline'].sum()
global_eta = total_steps / total_dist if total_dist > 0 else 0

# 4. Metrics & Breakdown
success_rate = df['success'].mean()
mean_steps = successful_df['steps'].mean()
std_steps = successful_df['steps'].std()

# Per-persona η calculation
persona_stats = df.groupby('persona').apply(lambda x: pd.Series({
    'success_rate': x['success'].mean(),
    'avg_steps': x.loc[x['success'], 'steps'].mean(),
    'avg_dist': x.loc[x['success'], 'dist_to_baseline'].mean(),
    'eta': x.loc[x['success'], 'steps'].sum() / x.loc[x['success'], 'dist_to_baseline'].sum()
})).reset_index()

# 5. Reporting
print("--- Evaluation Summary ---")
print(f"Success Rate:      {success_rate:.2%}")
print(f"Mean Steps:        {mean_steps:.2f} (±{std_steps:.2f})")
print(f"Efficiency (η):    {global_eta:.4f} steps/unit_dist")

print("\n--- Persona-Specific Efficiency ---")
print(persona_stats.to_string(index=False))

df = pd.read_csv('agent_judge_eval_diabetes_50.csv')

df['final_cpm_vector'] = df['final_cpm_vector'].apply(lambda x: np.array(ast.literal_eval(x)))

# Define Baseline
baseline = np.array([0.0, 0.5, 0.5, 0.5])

# 2. Distance and Success Filtering
df['dist_to_baseline'] = df['final_cpm_vector'].apply(lambda v: np.linalg.norm(v - baseline))
successful_df = df[df['success'] == True].copy()

# 3. Efficiency Score (η) Calculation
# Global η = Sum of Steps / Sum of Distances
total_steps = successful_df['steps'].sum()
total_dist = successful_df['dist_to_baseline'].sum()
global_eta = total_steps / total_dist if total_dist > 0 else 0

# 4. Metrics & Breakdown
success_rate = df['success'].mean()
mean_steps = successful_df['steps'].mean()
std_steps = successful_df['steps'].std()

# Per-persona η calculation
persona_stats = df.groupby('persona').apply(lambda x: pd.Series({
    'success_rate': x['success'].mean(),
    'avg_steps': x.loc[x['success'], 'steps'].mean(),
    'avg_dist': x.loc[x['success'], 'dist_to_baseline'].mean(),
    'eta': x.loc[x['success'], 'steps'].sum() / x.loc[x['success'], 'dist_to_baseline'].sum()
})).reset_index()

# 5. Reporting
print("--- Evaluation Summary ---")
print(f"Success Rate:      {success_rate:.2%}")
print(f"Mean Steps:        {mean_steps:.2f} (±{std_steps:.2f})")
print(f"Efficiency (η):    {global_eta:.4f} steps/unit_dist")

print("\n--- Persona-Specific Efficiency ---")
print(persona_stats.to_string(index=False))