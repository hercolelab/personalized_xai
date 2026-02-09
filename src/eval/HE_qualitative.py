import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

CSV_PATH = "PONTE_he.csv"
SEP = ";"

LIKERT_MAP = {
    "strongly disagree": 1,
    "disagree": 2,
    "agree": 3,
    "strongly agree": 4,
}

METRICS = ["cogn", "utiliy", "satisf"]

def normalize(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return LIKERT_MAP.get(s, None)

def extract_narratives(cols):
    keys = set()
    pat = re.compile(r"^(\d+)_(\d+)_(cogn|utiliy|satisf)")
    for c in cols:
        m = pat.match(c)
        if m:
            keys.add(f"{m.group(1)}_{m.group(2)}")
    return sorted(keys, key=lambda k: (int(k.split("_")[0]), int(k.split("_")[1])))

def persona_info(narr):
    r, p = map(int, narr.split("_"))
    if r in (1, 2):
        scenario = "Healthcare"
        persona = "Patient" if p == 1 else "Clinician"
    else:
        scenario = "Finance"
        persona = "Loan applicant" if p == 1 else "Bank officer"
    return scenario, persona

df = pd.read_csv(CSV_PATH, sep=SEP)
narratives = extract_narratives(df.columns)

rows = []

for _, r in df.iterrows():
    field = r.get("Field", None)
    for nar in narratives:
        scenario, persona = persona_info(nar)
        for m in METRICS:
            col = f"{nar}_{m}"
            if col not in df.columns:
                continue
            val = normalize(r[col])
            if val is None:
                continue
            rows.append({
                "scenario": scenario,
                "persona": persona,
                "field": field,
                "metric": m,
                "score": val
            })

long = pd.DataFrame(rows)

long["top2"] = (long["score"] >= 3).astype(int)

overall = (
    long.groupby("metric")
    .agg(proportion_agree=("top2", "mean"),
         n=("top2", "count"))
    .reset_index()
)

by_scenario = (
    long.groupby(["scenario", "metric"])
    .agg(proportion_agree=("top2", "mean"),
         n=("top2", "count"))
    .reset_index()
)

by_persona = (
    long.groupby(["scenario", "persona", "metric"])
    .agg(proportion_agree=("top2", "mean"),
         n=("top2", "count"))
    .reset_index()
)

by_field = (
    long.groupby(["field", "metric"])
    .agg(proportion_agree=("top2", "mean"),
         n=("top2", "count"))
    .reset_index()
    .sort_values(["metric", "field"])
)

by_persona_field = (
    long.groupby(["scenario", "persona", "field", "metric"])
    .agg(proportion_agree=("top2", "mean"),
         n=("top2", "count"))
    .reset_index()
    .sort_values(["scenario", "persona", "metric", "field"])
)

overall.to_csv("top2_overall.csv", index=False)
by_scenario.to_csv("top2_by_scenario.csv", index=False)
by_persona.to_csv("top2_by_persona.csv", index=False)
by_field.to_csv("top2_by_field.csv", index=False)
by_persona_field.to_csv("top2_by_persona_field.csv", index=False)

print("\n=== Overall ===")
print(overall)

print("\n=== By Scenario ===")
print(by_scenario)

print("\n=== By Persona ===")
print(by_persona)

print("\n=== By Field (overall) ===")
print(by_field)

print("\n=== By Persona and Field ===")
print(by_persona_field)

sns.set_theme(
    style="white", 
    font_scale=1.3,
    rc={
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold"
    }
)

label_map = {
    "cogn": "Linguistic Realization Quality",
    "utiliy": "Actionable Insights",
    "satisf": "Overall Satisfaction"
}

by_persona['metric'] = by_persona['metric'].map(label_map)
by_field['metric'] = by_field['metric'].map(label_map)

sns.set_theme(style="white", font_scale=1.1)
my_colors = sns.color_palette("Set2") 

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

sns.barplot(
    data=by_persona,
    x="persona",
    y="proportion_agree",
    hue="metric",
    palette=my_colors,
    errorbar=('ci', 95),
    capsize=.05,
    edgecolor="white",
    linewidth=1,
    ax=axes[0]
)

axes[0].set_title("by Use-case Persona", pad=15)
axes[0].set_ylabel("Proportion Agree/Strongly Agree", labelpad=10)
axes[0].set_xlabel("")
axes[0].grid(axis='y', linestyle='--', alpha=0.3)
axes[0].tick_params(axis='x', rotation=0) 
axes[0].legend_.remove()

sns.barplot(
    data=by_field,
    x="field",
    y="proportion_agree",
    hue="metric",
    palette=my_colors,
    errorbar=('ci', 95),
    capsize=.05,
    edgecolor="white",
    linewidth=1,
    ax=axes[1]
)

axes[1].set_title("by Field of Expertise", pad=15)
axes[1].set_ylabel("")
axes[1].set_xlabel("")
axes[1].grid(axis='y', linestyle='--', alpha=0.3)
axes[1].tick_params(axis='x', rotation=0)
axes[1].legend_.remove()

sns.despine()

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles, labels, 
    loc='lower center', 
    ncol=3, 
    frameon=False,
    bbox_to_anchor=(0.5, 0.005),
    prop={'weight': 'bold'}
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)

plt.show()