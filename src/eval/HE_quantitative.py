from __future__ import annotations

import re
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr

SURVEY_CSV = "PONTE_he.csv"
GROUND_TRUTH_CSV = "ground_truth.csv"
SEP_SURVEY = ";"
SEP_GT = ";"

STYLE_DIMS = ["tech", "verb", "depth", "action"]
N_PERM_OVERALL = 2000
N_PERM_NARR = 1000
SEED = 42

OUT_OVERALL = "overall_by_form.csv"
OUT_NARR = "narrative_by_form.csv"
OUT_DIM = "dimension_by_form.csv"
OUT_DIM_POOLED = "dimension_pooled_weighted.csv"
OUT_TESTS = "tests_overall.txt"
OUT_TESTS_NARR = "tests_narratives.csv"

CAT_TO_ORD = {"low": 0, "medium": 1, "high": 2}

def normalize_cat(x) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    if s in ("low", "l"):
        return "low"
    if s in ("medium", "med", "m"):
        return "medium"
    if s in ("high", "h"):
        return "high"
    return None

def gt_to_cat(x) -> str | None:
    c = normalize_cat(x)
    if c is not None:
        return c
    try:
        v = float(x)
    except Exception:
        return None
    if np.isclose(v, 0.2, atol=1e-6):
        return "low"
    if np.isclose(v, 0.5, atol=1e-6):
        return "medium"
    if np.isclose(v, 0.8, atol=1e-6):
        return "high"
    return None

def extract_narratives(columns) -> list[str]:
    keys = set()
    pat = re.compile(r"^(\d+)_(\d+)_(tech|verb|depth|action)\b", re.IGNORECASE)
    for col in columns:
        m = pat.match(col.strip())
        if m:
            keys.add(f"{m.group(1)}_{m.group(2)}")
    return sorted(keys, key=lambda k: (int(k.split("_")[0]), int(k.split("_")[1])))

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adj[i] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adj, 0, 1)
    return out

def safe_qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return np.nan
    if len(np.unique(np.concatenate([y_true, y_pred]))) < 2:
        return np.nan
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")

def compute_overall_metrics(long_df: pd.DataFrame) -> dict:
    d = long_df["human_ord"].to_numpy() - long_df["gt_ord"].to_numpy()
    abs_err = np.abs(d)
    mae = float(abs_err.mean())
    bias = float(d.mean())
    nmae = mae / 2.0
    nbias = bias / 2.0
    align = 1.0 - nmae
    qwk = safe_qwk(long_df["gt_ord"].to_numpy(), long_df["human_ord"].to_numpy())
    agg = (
        long_df.groupby(["narrative", "dimension"], as_index=False)
        .agg(human_mean=("human_ord", "mean"), gt=("gt_ord", "first"))
    )
    rho, p = spearmanr(agg["gt"].to_numpy(), agg["human_mean"].to_numpy())
    return {
        "n_ratings": int(len(long_df)),
        "n_units_narr_x_dim": int(len(agg)),
        "MAE_ord": mae,
        "Bias_ord": bias,
        "nMAE": nmae,
        "nBias": nbias,
        "Alignment": align,
        "QWK": float(qwk) if qwk is not None else np.nan,
        "Spearman_rho": float(rho),
        "Spearman_p": float(p),
    }

def compute_group_table(long_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    g = long_df.copy()
    g["diff"] = g["human_ord"] - g["gt_ord"]
    g["abs_err"] = (g["diff"]).abs()
    out = (
        g.groupby(group_cols, as_index=False)
        .agg(
            n_ratings=("abs_err", "size"),
            MAE_ord=("abs_err", "mean"),
            Bias_ord=("diff", "mean"),
        )
    )
    out["nMAE"] = out["MAE_ord"] / 2.0
    out["nBias"] = out["Bias_ord"] / 2.0
    out["Alignment"] = 1.0 - out["nMAE"]
    return out

def compute_dimension_pooled_weighted(dimension_by_form: pd.DataFrame) -> pd.DataFrame:
    def wavg(g: pd.DataFrame, col: str) -> float:
        w = g["n_ratings"].to_numpy()
        x = g[col].to_numpy()
        return float(np.sum(w * x) / np.sum(w))
    pooled = (
        dimension_by_form.groupby("dimension", as_index=False)
        .apply(lambda g: pd.Series({
            "n_ratings_total": int(g["n_ratings"].sum()),
            "nMAE_weighted": wavg(g, "nMAE"),
            "nBias_weighted": wavg(g, "nBias"),
            "MAE_ord_weighted": wavg(g, "MAE_ord"),
            "Bias_ord_weighted": wavg(g, "Bias_ord"),
        }))
        .reset_index(drop=True)
    )
    pooled["Alignment_weighted"] = 1.0 - pooled["nMAE_weighted"]
    dim_order = {d: i for i, d in enumerate(STYLE_DIMS)}
    pooled["dim_i"] = pooled["dimension"].map(dim_order)
    pooled = pooled.sort_values("dim_i").drop(columns="dim_i")
    return pooled

def perm_test_v1_v2(long_df: pd.DataFrame, n_perm: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    pid_form = long_df[["pid", "Form"]].drop_duplicates().set_index("pid")["Form"]
    pids = pid_form.index.to_numpy()
    forms = pid_form.to_numpy()
    def metrics_for_labels(form_map: dict[str, str]) -> dict:
        tmp = long_df.copy()
        tmp["Form_perm"] = tmp["pid"].map(form_map)
        m1 = compute_overall_metrics(tmp[tmp["Form_perm"] == "V1"].rename(columns={"Form_perm": "Form"}))
        m2 = compute_overall_metrics(tmp[tmp["Form_perm"] == "V2"].rename(columns={"Form_perm": "Form"}))
        return {
            "nMAE": m1["nMAE"] - m2["nMAE"],
            "nBias": m1["nBias"] - m2["nBias"],
            "QWK": m1["QWK"] - m2["QWK"],
            "Spearman_rho": m1["Spearman_rho"] - m2["Spearman_rho"],
        }
    obs_map = dict(zip(pids, forms))
    obs = metrics_for_labels(obs_map)
    stats = {k: [] for k in obs.keys()}
    for _ in range(n_perm):
        perm_forms = forms.copy()
        rng.shuffle(perm_forms)
        perm_map = dict(zip(pids, perm_forms))
        s = metrics_for_labels(perm_map)
        for k in stats:
            stats[k].append(s[k])
    res = {}
    for k in obs:
        arr = np.asarray(stats[k], dtype=float)
        p = (np.sum(np.abs(arr) >= abs(obs[k])) + 1) / (len(arr) + 1)
        res[k] = {"obs_diff_V1_minus_V2": float(obs[k]), "p_perm_two_sided": float(p)}
    return res

def perm_test_narrative_diffs(long_df: pd.DataFrame, n_perm: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pid_form = long_df[["pid", "Form"]].drop_duplicates().set_index("pid")["Form"]
    pids = pid_form.index.to_numpy()
    forms = pid_form.to_numpy()
    narratives = sorted(long_df["narrative"].unique(), key=lambda k: (int(k.split("_")[0]), int(k.split("_")[1])))
    def narrative_metrics(tmp: pd.DataFrame) -> pd.DataFrame:
        g = tmp.copy()
        g["diff"] = g["human_ord"] - g["gt_ord"]
        g["abs_err"] = g["diff"].abs()
        narr_form = (
            g.groupby(["narrative", "FormX"], as_index=False)
            .agg(MAE=("abs_err", "mean"), Bias=("diff", "mean"))
        )
        narr_form["nMAE"] = narr_form["MAE"] / 2.0
        narr_form["nBias"] = narr_form["Bias"] / 2.0
        p_mae = narr_form.pivot(index="narrative", columns="FormX", values="nMAE")
        p_bias = narr_form.pivot(index="narrative", columns="FormX", values="nBias")
        out = pd.DataFrame({"narrative": p_mae.index})
        out["diff_nMAE_V1_minus_V2"] = (p_mae.get("V1") - p_mae.get("V2")).values
        out["diff_nBias_V1_minus_V2"] = (p_bias.get("V1") - p_bias.get("V2")).values
        return out.set_index("narrative").reindex(narratives).reset_index()
    obs = long_df.copy()
    obs["FormX"] = obs["Form"]
    obs_tab = narrative_metrics(obs).set_index("narrative")
    perm_dists_mae = {nar: [] for nar in narratives}
    perm_dists_bias = {nar: [] for nar in narratives}
    for _ in range(n_perm):
        perm_forms = forms.copy()
        rng.shuffle(perm_forms)
        perm_map = dict(zip(pids, perm_forms))
        tmp = long_df.copy()
        tmp["FormX"] = tmp["pid"].map(perm_map)
        tab = narrative_metrics(tmp).set_index("narrative")
        for nar in narratives:
            perm_dists_mae[nar].append(tab.loc[nar, "diff_nMAE_V1_minus_V2"])
            perm_dists_bias[nar].append(tab.loc[nar, "diff_nBias_V1_minus_V2"])
    rows = []
    for nar in narratives:
        obs_mae = float(obs_tab.loc[nar, "diff_nMAE_V1_minus_V2"])
        obs_bias = float(obs_tab.loc[nar, "diff_nBias_V1_minus_V2"])
        arr_mae = np.asarray(perm_dists_mae[nar], dtype=float)
        arr_bias = np.asarray(perm_dists_bias[nar], dtype=float)
        p_mae = (np.sum(np.abs(arr_mae) >= abs(obs_mae)) + 1) / (len(arr_mae) + 1)
        p_bias = (np.sum(np.abs(arr_bias) >= abs(obs_bias)) + 1) / (len(arr_bias) + 1)
        rows.append({
            "narrative": nar,
            "diff_nMAE_V1_minus_V2": obs_mae,
            "p_perm_nMAE": float(p_mae),
            "diff_nBias_V1_minus_V2": obs_bias,
            "p_perm_nBias": float(p_bias),
        })
    out = pd.DataFrame(rows)
    out["p_perm_nMAE_FDR_BH"] = bh_fdr(out["p_perm_nMAE"].to_numpy())
    out["p_perm_nBias_FDR_BH"] = bh_fdr(out["p_perm_nBias"].to_numpy())
    return out

survey = pd.read_csv(SURVEY_CSV, sep=SEP_SURVEY)
gt = pd.read_csv(GROUND_TRUTH_CSV, sep=SEP_GT)

survey["Form"] = survey["Form"].astype(str).str.strip()
gt["Form"] = gt["Form"].astype(str).str.strip()
gt["narrative"] = gt["narrative"].astype(str).str.strip()

for d in STYLE_DIMS:
    gt[d] = gt[d].apply(gt_to_cat)

narratives = extract_narratives(survey.columns)
gt_any = gt.drop_duplicates(subset=["narrative"]).copy()
gt_long = gt_any.melt(id_vars=["narrative"], value_vars=STYLE_DIMS, var_name="dimension", value_name="gt_cat")
gt_long["gt_cat"] = gt_long["gt_cat"].map(lambda x: None if x is None else str(x).strip().lower())
gt_long["gt_ord"] = gt_long["gt_cat"].map(CAT_TO_ORD).astype("Float64")

rows = []
for pid, r in survey.iterrows():
    form = str(r["Form"]).strip()
    for nar in narratives:
        for dim in STYLE_DIMS:
            col = f"{nar}_{dim}"
            if col not in survey.columns:
                continue
            hc = normalize_cat(r[col])
            if hc is None:
                continue
            rows.append({"pid": str(pid), "Form": form, "narrative": nar, "dimension": dim, "human_ord": CAT_TO_ORD[hc]})

long_df = pd.DataFrame(rows)
long_df = long_df.merge(gt_long[["narrative", "dimension", "gt_ord"]], on=["narrative", "dimension"], how="left")
long_df = long_df.dropna(subset=["gt_ord"])
long_df["gt_ord"] = long_df["gt_ord"].astype(int)
long_df = long_df[long_df["Form"].isin(["V1", "V2"])].copy()

overall_rows = []
for form in ["V1", "V2"]:
    m = compute_overall_metrics(long_df[long_df["Form"] == form])
    m["Form"] = form
    overall_rows.append(m)

overall = pd.DataFrame(overall_rows)[["Form", "n_ratings", "n_units_narr_x_dim", "nMAE", "nBias", "Alignment", "QWK", "Spearman_rho", "Spearman_p", "MAE_ord", "Bias_ord"]]
overall.to_csv(OUT_OVERALL, index=False)

narr_tables = []
for form in ["V1", "V2"]:
    t = compute_group_table(long_df[long_df["Form"] == form], ["narrative"])
    t["Form"] = form
    narr_tables.append(t)

narrative_by_form = pd.concat(narr_tables, ignore_index=True)
narrative_by_form.to_csv(OUT_NARR, index=False)

dim_tables = []
for form in ["V1", "V2"]:
    t = compute_group_table(long_df[long_df["Form"] == form], ["dimension"])
    t["Form"] = form
    dim_tables.append(t)

dimension_by_form = pd.concat(dim_tables, ignore_index=True)
dimension_by_form.to_csv(OUT_DIM, index=False)

dimension_pooled = compute_dimension_pooled_weighted(dimension_by_form)
dimension_pooled.to_csv(OUT_DIM_POOLED, index=False)

tests_overall = perm_test_v1_v2(long_df, n_perm=N_PERM_OVERALL, seed=SEED)
tests_narr = perm_test_narrative_diffs(long_df, n_perm=N_PERM_NARR, seed=SEED + 7)
tests_narr.to_csv(OUT_TESTS_NARR, index=False)

lines = []
lines.append("=== Overall alignment metrics (per Form) ===")
lines.append(overall.to_string(index=False))
lines.append("\n=== Dimension-level metrics (per Form) ===")
lines.append(dimension_by_form.to_string(index=False))
lines.append("\n=== Dimension-level pooled metrics across forms (weighted by n_ratings) ===")
lines.append(dimension_pooled.to_string(index=False))
lines.append("\n=== Participant-label permutation tests: difference (V1 - V2) ===")
for k in ["nMAE", "nBias", "QWK", "Spearman_rho"]:
    lines.append(f"{k:>12s}: obs_diff={tests_overall[k]['obs_diff_V1_minus_V2']:+.6f}  p_perm={tests_overall[k]['p_perm_two_sided']:.6g}")

with open(OUT_TESTS, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\n".join(lines))