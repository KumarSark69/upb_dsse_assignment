"""
WEEK 2 — Complete Pipeline: LDA + BERTopic Topic Modeling
Answers Research Question 1.

Usage:
    python week2_complete.py

Requirements:
    pip install scikit-learn gensim matplotlib seaborn bertopic sentence-transformers

IMPORTANT — folder structure:
    Place week2_complete.py in the SAME folder as week1_complete.py.
    Both scripts read from and write to the same data/, plots/, results/ folders:

    your_project/
    ├── week1_complete.py       ← run this first
    ├── week2_complete.py       ← run this second
    ├── Issues.xlsx
    ├── data/
    │   ├── issues_raw.json         (created by week1)
    │   ├── issues_processed.json   (created by week1)
    │   ├── vocabulary.csv          (created by week1)
    │   ├── dtm.npz                 (created by week1)
    │   └── dtm_feature_names.json  (created by week1)
    ├── plots/                  (week1 plots already here; week2 adds more)
    └── results/                (week2 adds rq1_topics.csv here)
"""

import json, re, numpy as np, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import Counter
from scipy import sparse

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd

from sklearn.decomposition import LatentDirichletAllocation
from gensim import corpora
from gensim.models import CoherenceModel

# ── Same paths as week1_complete.py ──────────────────────────────────────────
DATA_DIR    = Path("data");    DATA_DIR.mkdir(exist_ok=True)
PLOTS_DIR   = Path("plots");   PLOTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("results"); RESULTS_DIR.mkdir(exist_ok=True)

COLORS = plt.cm.tab10.colors


# =============================================================================
# Helpers
# =============================================================================

def load_week1():
    """
    Load the three files produced by week1_complete.py.
    All three must exist in data/ before running week2.
    """
    required = [
        DATA_DIR / "issues_processed.json",
        DATA_DIR / "dtm.npz",
        DATA_DIR / "dtm_feature_names.json",
    ]
    missing = [str(f) for f in required if not f.exists()]
    if missing:
        print("\n[ERROR] Week 1 output files not found:")
        for m in missing:
            print(f"  missing → {m}")
        print("\nPlease run week1_complete.py first, then re-run this script.")
        raise SystemExit(1)

    with open(DATA_DIR / "issues_processed.json") as f:
        issues = json.load(f)
    dtm = sparse.load_npz(str(DATA_DIR / "dtm.npz"))
    with open(DATA_DIR / "dtm_feature_names.json") as f:
        vocab = json.load(f)
    return issues, dtm, vocab


def get_top_words(model, vocab, n=15):
    return [[vocab[i] for i in t.argsort()[:-n-1:-1]] for t in model.components_]


# =============================================================================
# STEP 1 — Find best number of topics via coherence score
# =============================================================================

def step1_coherence(dtm, vocab, token_lists):
    print("\n── STEP 1: Coherence scoring (n = 3 to 10) ──")
    dictionary = corpora.Dictionary(token_lists)
    N_RANGE = list(range(3, 11))
    scores  = []

    for n in N_RANGE:
        lda = LatentDirichletAllocation(
            n_components=n, doc_topic_prior=0.01, topic_word_prior=0.01,
            max_iter=50, learning_method="batch", random_state=42
        )
        lda.fit(dtm)
        tw = [[w for w in words if w in dictionary.token2id]
              for words in get_top_words(lda, vocab, 15)]
        score = CoherenceModel(topics=tw, texts=token_lists,
                               dictionary=dictionary, coherence="c_v").get_coherence()
        scores.append(score)
        print(f"  n={n}  coherence={score:.4f}")

    best_n = N_RANGE[int(np.argmax(scores))]
    print(f"\n  Best n_topics = {best_n}  (coherence = {max(scores):.4f})")

    with open(DATA_DIR / "coherence_scores.json", "w") as f:
        json.dump({"n_range": N_RANGE, "scores": scores, "best_n": best_n}, f)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(N_RANGE, scores, marker="o", color="steelblue", linewidth=2.5, markersize=7)
    ax.axvline(best_n, color="tomato", linestyle="--", linewidth=2, label=f"Best n = {best_n}")
    for n, s in zip(N_RANGE, scores):
        ax.annotate(f"{s:.3f}", (n, s), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel("Number of Topics", fontsize=12)
    ax.set_ylabel("Coherence Score (C_V)", fontsize=12)
    ax.set_title("LDA Coherence Score vs Number of Topics", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "lda_coherence.png", dpi=150)
    plt.close()
    print("  ✓ Saved plots/lda_coherence.png")
    return best_n, dictionary


# =============================================================================
# STEP 2 — Train final LDA and generate all LDA plots
# =============================================================================

def step2_lda(dtm, vocab, token_lists, best_n, issues):
    print(f"\n── STEP 2: Training final LDA ({best_n} topics) ──")

    lda = LatentDirichletAllocation(
        n_components=best_n, doc_topic_prior=0.01, topic_word_prior=0.01,
        max_iter=100, learning_method="batch", random_state=42
    )
    lda.fit(dtm)

    top_words    = get_top_words(lda, vocab, 15)
    doc_topic    = lda.transform(dtm)
    dominant     = doc_topic.argmax(axis=1).tolist()
    topic_counts = Counter(dominant)
    avg_props    = doc_topic.mean(axis=0).tolist()

    print("\n  Topics found:")
    for i, words in enumerate(top_words):
        print(f"  Topic {i} ({topic_counts.get(i,0)} issues): {', '.join(words[:8])}")

    # ── Plots ────────────────────────────────────────────────────────────────
    # Top words grid
    cols, rows = 4, (best_n + 3) // 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4.5, rows*4))
    axes = axes.flatten()
    for i, words in enumerate(top_words):
        w12 = words[:12]
        ax  = axes[i]
        ax.barh(range(len(w12)), [len(w12)-j for j in range(len(w12))],
                color=COLORS[i % 10], alpha=0.75, edgecolor="white")
        ax.set_yticks(range(len(w12))); ax.set_yticklabels(w12, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(f"Topic {i}  ({topic_counts.get(i,0)} issues)",
                     fontsize=10, fontweight="bold", color=COLORS[i % 10])
        ax.set_xticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
    for j in range(i+1, len(axes)): axes[j].set_visible(False)
    fig.suptitle("LDA — Top 12 Words per Topic", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "lda_top_words.png", dpi=150); plt.close()

    # Issues per topic
    fig, ax = plt.subplots(figsize=(9, 4))
    cnts = [topic_counts.get(i, 0) for i in range(best_n)]
    bars = ax.bar([f"Topic {i}" for i in range(best_n)], cnts,
                  color=[COLORS[i%10] for i in range(best_n)], alpha=0.85, edgecolor="white")
    for bar, c in zip(bars, cnts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                str(c), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Number of Issues", fontsize=12)
    ax.set_title("Number of Issues per LDA Topic (RQ1)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, max(cnts)+2)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "lda_issues_per_topic.png", dpi=150); plt.close()

    # Average topic proportions
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar([f"Topic {i}" for i in range(best_n)], avg_props,
                  color=[COLORS[i%10] for i in range(best_n)], alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, avg_props):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Average Proportion per Issue", fontsize=12)
    ax.set_title("Average Topic Proportions Across All Issues", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "lda_topic_proportions.png", dpi=150); plt.close()

    # Heatmap of doc-topic dist (first 30 issues)
    sample = doc_topic[:30]
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(sample, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=[f"T{i}" for i in range(best_n)],
                yticklabels=[f"Issue {i}" for i in range(30)],
                linewidths=0.3, ax=ax, cbar_kws={"label": "Topic proportion"})
    ax.set_xlabel("LDA Topic", fontsize=11); ax.set_ylabel("Issue", fontsize=11)
    ax.set_title("Topic Proportions per Issue (first 30)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "lda_doc_topic_heatmap.png", dpi=150); plt.close()

    print("  ✓ Saved 4 LDA plots")

    # Save results
    lda_results = {
        "n_topics": best_n,
        "topics": [{"topic_id": i, "top_words": top_words[i],
                    "num_issues": topic_counts.get(i, 0)} for i in range(best_n)],
        "doc_topic_dist": doc_topic.tolist(),
        "dominant_topic_per_issue": dominant,
        "avg_topic_proportions": avg_props,
    }
    with open(DATA_DIR / "lda_results.json", "w") as f:
        json.dump(lda_results, f, indent=2)

    # Add LDA topic to each issue
    for i, issue in enumerate(issues):
        issue["lda_topic"] = dominant[i]
    with open(DATA_DIR / "issues_with_topics.json", "w") as f:
        json.dump(issues, f, indent=2)

    print("  ✓ Saved data/lda_results.json")
    return lda_results


# =============================================================================
# STEP 3 — BERTopic
# =============================================================================

def step3_bertopic(issues):
    print("\n── STEP 3: Running BERTopic ──")
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import CountVectorizer
    except ImportError:
        print("  [SKIP] BERTopic not installed.")
        print("  Install with: pip install bertopic sentence-transformers")
        print("  Then re-run this script — LDA results are already saved.")
        return None

    docs = [f"{i['summary']} {i['description']}".strip() or i["summary"]
            for i in issues]

    custom_stop = [
        "issue","fix","test","use","add","get","set","new","null","true","false",
        "http","https","org","com","apache","yarn","hadoop","lucene","tika",
        "jclouds","also","would","could","may","might","make","one","need",
        "see","please","note","code","line","change","update","user","using"
    ]
    vectorizer = CountVectorizer(stop_words=custom_stop, ngram_range=(1, 2), min_df=2)

    topic_model = BERTopic(
        embedding_model=SentenceTransformer("all-MiniLM-L6-v2"),
        vectorizer_model=vectorizer,
        nr_topics="auto",
        calculate_probabilities=True,
        verbose=False
    )
    topics, probs = topic_model.fit_transform(docs)

    topic_info = topic_model.get_topic_info()
    print(f"  BERTopic found {len(topic_info)-1} topics")
    print(topic_info[topic_info["Topic"] != -1].to_string(index=False))

    # Plot: top words per topic
    real_topics = topic_info[topic_info["Topic"] != -1].head(12)
    cols = 3; rows = (len(real_topics) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*3.5))
    axes = axes.flatten()
    for i, (_, row) in enumerate(real_topics.iterrows()):
        tid = row["Topic"]
        ws  = topic_model.get_topic(tid) or []
        words, scores = zip(*ws[:10]) if ws else ([], [])
        ax = axes[i]
        ax.barh(list(words)[::-1], list(scores)[::-1], color=COLORS[i%10], alpha=0.8)
        ax.set_title(f"Topic {tid}  ({row['Count']} issues)", fontsize=9, fontweight="bold")
        ax.set_xlabel("c-TF-IDF", fontsize=8)
        for sp in ax.spines.values(): sp.set_visible(False)
    for j in range(i+1, len(axes)): axes[j].set_visible(False)
    fig.suptitle("BERTopic — Top Words per Topic", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "bertopic_top_words.png", dpi=150); plt.close()

    # Plot: issues per topic
    real = topic_info[topic_info["Topic"] != -1].sort_values("Count", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(real["Topic"].astype(str), real["Count"],
           color=[COLORS[i%10] for i in range(len(real))], alpha=0.85, edgecolor="white")
    ax.set_xlabel("BERTopic ID"); ax.set_ylabel("Number of Issues")
    ax.set_title("Number of Issues per BERTopic (RQ1)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "bertopic_issues_per_topic.png", dpi=150); plt.close()

    print("  ✓ Saved 2 BERTopic plots")

    bert_results = {
        "num_topics": int(real["Topic"].max() + 1),
        "topic_info": topic_info.to_dict("records"),
        "topics": {str(t): topic_model.get_topic(t)
                   for t in topic_info["Topic"] if t != -1},
        "doc_topics": [int(t) for t in topics],
    }
    with open(DATA_DIR / "bertopic_results.json", "w") as f:
        json.dump(bert_results, f, indent=2)
    print("  ✓ Saved data/bertopic_results.json")
    return bert_results


# =============================================================================
# STEP 4 — RQ1 summary table
# =============================================================================

def step4_rq1_table(lda_results, bert_results=None):
    print("\n── STEP 4: Building RQ1 summary table ──")

    rows = []
    for t in lda_results["topics"]:
        rows.append({
            "Model":          "LDA",
            "Topic ID":       t["topic_id"],
            "# Issues":       t["num_issues"],
            "Avg Proportion": round(lda_results["avg_topic_proportions"][t["topic_id"]], 3),
            "Top 10 Keywords": ", ".join(t["top_words"][:10]),
        })

    if bert_results:
        for info in bert_results["topic_info"]:
            tid = info["Topic"]
            if tid == -1: continue
            words_scores = bert_results["topics"].get(str(tid), [])
            keywords = ", ".join(w for w, _ in words_scores[:10])
            rows.append({
                "Model":           "BERTopic",
                "Topic ID":        tid,
                "# Issues":        info["Count"],
                "Avg Proportion":  "-",
                "Top 10 Keywords": keywords,
            })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "rq1_topics.csv", index=False)
    print("  ✓ Saved results/rq1_topics.csv")
    print("\n  LDA Topics (RQ1):")
    print(df[df["Model"]=="LDA"][["Topic ID","# Issues","Top 10 Keywords"]].to_string(index=False))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  WEEK 2 — Topic Modeling (LDA + BERTopic)")
    print("=" * 55)

    issues, dtm, vocab = load_week1()
    token_lists = [i["tokens"] for i in issues]
    print(f"Loaded: {len(issues)} issues, DTM {dtm.shape}, vocab {len(vocab)}")

    best_n, dictionary = step1_coherence(dtm, vocab, token_lists)
    lda_results        = step2_lda(dtm, vocab, token_lists, best_n, issues)
    bert_results       = step3_bertopic(issues)
    step4_rq1_table(lda_results, bert_results)

    print("\n" + "=" * 55)
    print("  WEEK 2 COMPLETE")
    print(f"  LDA topics found:     {lda_results['n_topics']}")
    if bert_results:
        print(f"  BERTopic topics:      {bert_results['num_topics']}")
    print("  All outputs in:  data/  plots/  results/")
    print("  Ready for Week 3 (statistical analysis)")
    print("=" * 55)
