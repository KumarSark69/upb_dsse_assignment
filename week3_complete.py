"""
WEEK 3 — Significance Tests on Topic Characteristics and Co-occurrences
Answers Research Question 2 and Research Question 3.

Usage:
    python week3_complete.py

Requirements:
    pip install pandas numpy scipy statsmodels matplotlib seaborn

IMPORTANT — folder structure:
    Place week3_complete.py in the SAME folder as week1_complete.py and week2_complete.py.
    All scripts share the same data/, plots/, results/ folders:

    assignment/
    ├── week1_complete.py
    ├── week2_complete.py
    ├── week3_complete.py       ← this script
    ├── Issues.xlsx
    ├── data/
    │   ├── issues_processed.json       (created by week1)
    │   ├── issues_with_topics.json     (created by week2)  ← loaded here
    │   ├── lda_results.json            (created by week2)  ← loaded here
    │   └── bertopic_results.json       (created by week2)  ← loaded here (optional)
    ├── plots/                  (week3 adds RQ2 + RQ3 plots here)
    └── results/                (week3 adds rq2_stats.csv + rq3_stats.csv here)
"""

import json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import kruskal, mannwhitneyu, chi2_contingency
from statsmodels.stats.multitest import multipletests

# ── Same paths as week1_complete.py and week2_complete.py ────────────────────
DATA_DIR    = Path("data");    DATA_DIR.mkdir(exist_ok=True)
PLOTS_DIR   = Path("plots");   PLOTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("results"); RESULTS_DIR.mkdir(exist_ok=True)

COLORS = plt.cm.tab10.colors

# Numeric characteristics to analyse per topic (RQ2)
CHAR_COLS = ["num_comments", "num_human_comments", "num_attachments", "desc_length"]
CHAR_LABELS = {
    "num_comments":       "Number of Comments",
    "num_human_comments": "Number of Human Comments",
    "num_attachments":    "Number of Attachments",
    "desc_length":        "Description Length (chars)",
}


# =============================================================================
# Load data produced by week2_complete.py
# =============================================================================

def load_week2():
    """
    Load the files produced by week2_complete.py.
    issues_with_topics.json and lda_results.json are required.
    bertopic_results.json is optional (only used if BERTopic ran successfully).
    """
    required = [
        DATA_DIR / "issues_with_topics.json",
        DATA_DIR / "lda_results.json",
    ]
    missing = [str(f) for f in required if not f.exists()]
    if missing:
        print("\n[ERROR] Week 2 output files not found:")
        for m in missing:
            print(f"  missing → {m}")
        print("\nPlease run week2_complete.py first, then re-run this script.")
        raise SystemExit(1)

    with open(DATA_DIR / "issues_with_topics.json") as f:
        issues = json.load(f)
    with open(DATA_DIR / "lda_results.json") as f:
        lda = json.load(f)

    bert = None
    bert_path = DATA_DIR / "bertopic_results.json"
    if bert_path.exists():
        with open(bert_path) as f:
            bert = json.load(f)
        print("  BERTopic results found — RQ3 will include BERTopic co-occurrences.")
    else:
        print("  No bertopic_results.json found — RQ3 will use LDA topics only.")

    return issues, lda, bert


def build_dataframe(issues, lda, bert):
    """Combine issue metadata, LDA topic, and BERTopic topic into one DataFrame."""
    lda_dominant  = lda["dominant_topic_per_issue"]
    bert_dominant = bert["doc_topics"] if bert else [None] * len(issues)

    rows = []
    for i, issue in enumerate(issues):
        rows.append({
            "issue_id":           issue["issue_id"],
            "project":            issue["project"],
            "dd1":                bool(issue["dd1"]),
            "dd2":                bool(issue["dd2"]),
            "dd3":                bool(issue["dd3"]),
            "num_comments":       int(issue.get("num_comments", 0)),
            "num_human_comments": int(issue.get("num_human_comments", 0)),
            "num_attachments":    int(issue.get("num_attachments", 0)),
            "desc_length":        len(issue.get("description", "")),
            "issue_type":         issue.get("issue_type", "Unknown"),
            "lda_topic":          int(lda_dominant[i]),
            "bert_topic":         int(bert_dominant[i]) if bert_dominant[i] is not None else None,
        })
    return pd.DataFrame(rows)


# =============================================================================
# STEP 1 — RQ2: Box plots of issue characteristics per topic
# =============================================================================

def step1_rq2_boxplots(df):
    print("\n── STEP 1: RQ2 — Box plots per topic ──")

    for topic_col, label in [("lda_topic", "LDA"), ("bert_topic", "BERTopic")]:
        if topic_col == "bert_topic" and df["bert_topic"].isnull().all():
            print(f"  [SKIP] No BERTopic assignments found.")
            continue

        plot_df = df.copy()
        if topic_col == "bert_topic":
            plot_df = plot_df[plot_df["bert_topic"] != -1].dropna(subset=["bert_topic"])
            plot_df["bert_topic"] = plot_df["bert_topic"].astype(int)

        topics = sorted(plot_df[topic_col].unique())
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for ax, col in zip(axes, CHAR_COLS):
            data = [plot_df[plot_df[topic_col] == t][col].dropna().values for t in topics]
            bp   = ax.boxplot(data, patch_artist=True, showfliers=False)
            for patch, color in zip(bp["boxes"], COLORS):
                patch.set_facecolor(color); patch.set_alpha(0.7)
            ax.set_xticks(range(1, len(topics) + 1))
            ax.set_xticklabels([f"T{t}" for t in topics], fontsize=9)
            ax.set_title(CHAR_LABELS[col], fontsize=11)
            ax.set_xlabel("Topic"); ax.grid(axis="y", alpha=0.3)

        fig.suptitle(f"RQ2 — Issue Characteristics per {label} Topic",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        path = PLOTS_DIR / f"rq2_boxplots_{label.lower()}.png"
        fig.savefig(path, dpi=150); plt.close()
        print(f"  ✓ Saved plots/rq2_boxplots_{label.lower()}.png")


# =============================================================================
# STEP 2 — RQ2: Kruskal-Wallis + pairwise Mann-Whitney U significance tests
# =============================================================================

def step2_rq2_stats(df):
    print("\n── STEP 2: RQ2 — Significance tests ──")
    all_records = []

    for topic_col, model_label in [("lda_topic", "LDA"), ("bert_topic", "BERTopic")]:
        if topic_col == "bert_topic" and df["bert_topic"].isnull().all():
            continue

        sub = df.copy()
        if topic_col == "bert_topic":
            sub = sub[sub["bert_topic"] != -1].dropna(subset=["bert_topic"])
            sub["bert_topic"] = sub["bert_topic"].astype(int)

        topics = sorted(sub[topic_col].unique())

        for col in CHAR_COLS:
            groups = [sub[sub[topic_col] == t][col].dropna().values for t in topics]

            # Overall Kruskal-Wallis test
            try:
                stat, p_kw = kruskal(*[g for g in groups if len(g) > 0])
            except Exception:
                stat, p_kw = float("nan"), float("nan")

            all_records.append({
                "model": model_label, "variable": col,
                "test": "Kruskal-Wallis (overall)",
                "topic_a": "all", "topic_b": "all",
                "statistic": round(stat, 4), "p_value": round(p_kw, 6),
                "p_adj_bonferroni": "-",
                "significant_p05": p_kw < 0.05 if not np.isnan(p_kw) else False,
            })

            # Pairwise Mann-Whitney U with Bonferroni correction
            pairs    = list(combinations(range(len(topics)), 2))
            p_values = []
            for ia, ib in pairs:
                if len(groups[ia]) > 0 and len(groups[ib]) > 0:
                    _, p = mannwhitneyu(groups[ia], groups[ib], alternative="two-sided")
                    p_values.append(p)
                else:
                    p_values.append(1.0)

            if p_values:
                _, p_adj, _, _ = multipletests(p_values, method="bonferroni")
                for (ia, ib), p_raw, p_corr in zip(pairs, p_values, p_adj):
                    all_records.append({
                        "model": model_label, "variable": col,
                        "test": "Mann-Whitney U (pairwise)",
                        "topic_a": topics[ia], "topic_b": topics[ib],
                        "statistic": "-",
                        "p_value": round(p_raw, 6),
                        "p_adj_bonferroni": round(p_corr, 6),
                        "significant_p05": p_corr < 0.05,
                    })

    rq2_df = pd.DataFrame(all_records)
    rq2_df.to_csv(RESULTS_DIR / "rq2_stats.csv", index=False)
    print("  ✓ Saved results/rq2_stats.csv")

    sig = rq2_df[rq2_df["significant_p05"] == True]
    print(f"  Significant results (p < 0.05 after Bonferroni): {len(sig)} / {len(rq2_df)}")
    if len(sig) > 0:
        print(sig[["model","variable","test","topic_a","topic_b",
                    "p_value","p_adj_bonferroni"]].head(10).to_string(index=False))
    return rq2_df


# =============================================================================
# STEP 3 — RQ3: Co-occurrence heatmaps + chi-square tests
# =============================================================================

def _chi2_result(ct, label):
    try:
        chi2, p, dof, _ = chi2_contingency(ct.values)
        return {"comparison": label, "chi2": round(chi2, 4),
                "p_value": round(p, 8), "dof": dof,
                "significant_p05": p < 0.05}
    except Exception as e:
        return {"comparison": label, "chi2": "-", "p_value": "-",
                "dof": "-", "significant_p05": False, "note": str(e)}


def _heatmap(ct, title, filename):
    ct_norm = ct.div(ct.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(max(7, ct.shape[1] * 0.9),
                                    max(5, ct.shape[0] * 0.6)))
    sns.heatmap(ct_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.4, ax=ax, cbar_kws={"label": "Row proportion"})
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=150); plt.close()
    print(f"  ✓ Saved plots/{filename}")


def step3_rq3_cooccurrence(df, bert):
    print("\n── STEP 3: RQ3 — Co-occurrence significance tests ──")
    records = []

    # Use clean copy without BERTopic outliers
    df_clean = df.copy()
    if bert:
        df_clean = df_clean[df_clean["bert_topic"] != -1].dropna(subset=["bert_topic"])
        df_clean["bert_topic"] = df_clean["bert_topic"].astype(int)

    # ── 1. LDA topics × BERTopic topics ──────────────────────────────────────
    if bert and "bert_topic" in df_clean.columns:
        ct = pd.crosstab(df_clean["lda_topic"], df_clean["bert_topic"])
        _heatmap(ct, "Co-occurrence: LDA Topics × BERTopic Topics",
                 "rq3_lda_vs_bert.png")
        records.append(_chi2_result(ct, "LDA × BERTopic"))

    # ── 2. LDA topics × design decision types ────────────────────────────────
    for dd in ["dd1", "dd2", "dd3"]:
        tmp = df.copy()
        tmp[dd] = tmp[dd].astype(int)
        ct = pd.crosstab(tmp["lda_topic"], tmp[dd])
        _heatmap(ct, f"Co-occurrence: LDA Topics × {dd.upper()}",
                 f"rq3_lda_vs_{dd}.png")
        records.append(_chi2_result(ct, f"LDA × {dd.upper()}"))

    # ── 3. BERTopic topics × design decision types ───────────────────────────
    if bert and "bert_topic" in df_clean.columns:
        for dd in ["dd1", "dd2", "dd3"]:
            tmp = df_clean.copy()
            tmp[dd] = tmp[dd].astype(int)
            ct = pd.crosstab(tmp["bert_topic"], tmp[dd])
            _heatmap(ct, f"Co-occurrence: BERTopic Topics × {dd.upper()}",
                     f"rq3_bert_vs_{dd}.png")
            records.append(_chi2_result(ct, f"BERTopic × {dd.upper()}"))

    rq3_df = pd.DataFrame(records)
    rq3_df.to_csv(RESULTS_DIR / "rq3_stats.csv", index=False)
    print("  ✓ Saved results/rq3_stats.csv")
    print("\n  Chi-square results:")
    print(rq3_df.to_string(index=False))
    return rq3_df


# =============================================================================
# STEP 4 — Print final summary
# =============================================================================

def step4_summary(rq2_df, rq3_df):
    print("\n── STEP 4: Summary ──")
    sig_rq2 = rq2_df[rq2_df["significant_p05"] == True]
    sig_rq3 = rq3_df[rq3_df["significant_p05"] == True] if "significant_p05" in rq3_df else pd.DataFrame()
    print(f"""
  RQ2 — Characteristics per topic:
    Tests run:    {len(rq2_df)}
    Significant:  {len(sig_rq2)}  (p < 0.05, Bonferroni corrected)
    → See results/rq2_stats.csv
    → See plots/rq2_boxplots_lda.png
    → See plots/rq2_boxplots_bertopic.png  (if BERTopic ran)

  RQ3 — Co-occurrence tests:
    Tests run:    {len(rq3_df)}
    Significant:  {len(sig_rq3)}  (p < 0.05)
    → See results/rq3_stats.csv
    → See plots/rq3_*.png
    """)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  WEEK 3 — Statistical Analysis (RQ2 + RQ3)")
    print("=" * 55)

    print("\nLoading Week 2 outputs ...")
    issues, lda, bert = load_week2()
    df = build_dataframe(issues, lda, bert)
    print(f"  {len(df)} issues loaded into dataframe")
    print(f"  LDA topics:      {sorted(df['lda_topic'].unique())}")
    if bert:
        print(f"  BERTopic topics: {sorted(df['bert_topic'].dropna().unique())}")

    step1_rq2_boxplots(df)
    rq2_df = step2_rq2_stats(df)
    rq3_df = step3_rq3_cooccurrence(df, bert)
    step4_summary(rq2_df, rq3_df)

    print("=" * 55)
    print("  WEEK 3 COMPLETE")
    print("  Plots  → plots/rq2_*.png  and  plots/rq3_*.png")
    print("  Stats  → results/rq2_stats.csv  and  results/rq3_stats.csv")
    print("=" * 55)
