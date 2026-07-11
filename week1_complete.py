"""
Run this single script to execute all of Week 1.

Usage:
    python week1_complete.py

Requirements:
    pip install requests pandas openpyxl nltk scipy

Before running:
    1. Place Issues.xlsx in the same folder as this script
    2. Run on your own machine (needs access to issues.apache.org)
"""

import json, re, time, requests
from pathlib import Path
from collections import Counter

import nltk
import pandas as pd
import numpy as np
from scipy import sparse

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = Path("data");  DATA_DIR.mkdir(exist_ok=True)
PLOTS_DIR = Path("plots"); PLOTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("results"); RESULTS_DIR.mkdir(exist_ok=True)

# ── Bot authors to filter out ─────────────────────────────────────────────────
BOT_AUTHORS = {
    "Hive QA","cnsgithub","TrafficServer Bot","Mail Delivery Subsystem",
    "ASF Subversion and Git Services","Hadoop QA","QABot from busbey",
    "Thomas Smets - A3 SYSTEM","ATLAS QA","m","Flink Jira Bot",
    "ASF IRC Bot","Beam JIRA Bot","tester","Mahout QA","Laurent Chabot",
    "TezQA","FAURE SYSTEMS","SentryQA","Bug Reporter","Chris Chabot",
    "Ignite TC Bot","asapsystems","rangerqa","Flume QA","Knox QA",
    "Giraph QA","Jerry Chabot","Sqoop QA Bot","apache@tingo.org",
    "GitHub Import","Tajo QA","Hudson","ASF GitHub Bot","genericqa"
}

# ── Ontology replacement map ──────────────────────────────────────────────────
ONTOLOGY_MAP = {
    "machine":"COMPONENT","service":"COMPONENT","module":"COMPONENT",
    "class":"COMPONENT","method":"COMPONENT","function":"COMPONENT",
    "handler":"COMPONENT","manager":"COMPONENT","scheduler":"COMPONENT",
    "executor":"COMPONENT","worker":"COMPONENT","thread":"COMPONENT",
    "container":"COMPONENT","node":"COMPONENT","server":"COMPONENT",
    "client":"COMPONENT","plugin":"COMPONENT","library":"COMPONENT",
    "send":"CONNECTOR","write":"CONNECTOR","retrieve":"CONNECTOR",
    "fetch":"CONNECTOR","receive":"CONNECTOR","publish":"CONNECTOR",
    "call":"CONNECTOR","invoke":"CONNECTOR","request":"CONNECTOR",
    "response":"CONNECTOR","connect":"CONNECTOR","transfer":"CONNECTOR",
    "message":"DATA","object":"DATA","dump":"DATA","file":"DATA",
    "log":"DATA","record":"DATA","packet":"DATA","buffer":"DATA",
    "queue":"DATA","stream":"DATA",
    "layer":"SOLUTION","pattern":"SOLUTION","replication":"SOLUTION",
    "authentication":"SOLUTION","cache":"SOLUTION","proxy":"SOLUTION",
    "migration":"SOLUTION","refactor":"SOLUTION","upgrade":"SOLUTION",
    "performance":"QUALITY","security":"QUALITY","scalability":"QUALITY",
    "reliability":"QUALITY","availability":"QUALITY","latency":"QUALITY",
    "throughput":"QUALITY","fault":"QUALITY","efficiency":"QUALITY",
}

STOP_WORDS = set(stopwords.words("english")) | {
    "issue","fix","test","use","add","get","set","new","null","true","false",
    "http","https","org","com","apache","yarn","hadoop","lucene","tika",
    "jclouds","mapreduce","also","would","could","may","might","like",
    "make","one","need","see","please","note","code","line","change",
    "update","user","using","used","case","two","hi","hello","thanks",
}

JIRA_BASE = "https://issues.apache.org/jira/rest/api/2"
HEADERS = {"User-Agent": "research-bot/1.0", "Accept": "application/json"}

# =============================================================================
# STEP 1 — Download from Jira
# =============================================================================

def load_issue_ids(xlsx_path="Issues.xlsx"):
    xl = pd.ExcelFile(xlsx_path)
    rows = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        df.columns = ["issue_id", "decisions"]
        df["project"] = sheet
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    split = df["decisions"].str.split(" ", expand=True)
    df["dd1"] = split[0].str.lower() == "true"
    df["dd2"] = split[1].str.lower() == "true"
    df["dd3"] = split[2].str.lower() == "true"
    df.drop(columns=["decisions"], inplace=True)
    return df.to_dict("records")

def fetch_issue(issue_id):
    url = f"{JIRA_BASE}/issue/{issue_id}"
    params = {"fields": "summary,description,issuetype,status,comment,parent,attachment"}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [WARN] {issue_id}: {e}")
        return None

def parse_issue(raw, meta):
    fields = raw.get("fields", {})
    comments_raw = fields.get("comment", {}).get("comments", [])
    human_comments = [
        c["body"] for c in comments_raw
        if c.get("author", {}).get("displayName", "") not in BOT_AUTHORS
    ]
    parent = fields["parent"]["key"] if "parent" in fields else None
    return {
        "issue_id":           raw["key"],
        "project":            meta["project"],
        "dd1":                meta["dd1"],
        "dd2":                meta["dd2"],
        "dd3":                meta["dd3"],
        "summary":            fields.get("summary") or "",
        "description":        fields.get("description") or "",
        "issue_type":         fields.get("issuetype", {}).get("name", ""),
        "status":             fields.get("status", {}).get("name", ""),
        "num_comments":       len(comments_raw),
        "num_human_comments": len(human_comments),
        "num_attachments":    len(fields.get("attachment", [])),
        "parent":             parent,
        "human_comments":     human_comments,
    }

def step1_download():
    print("\n── STEP 1: Downloading issues from Jira ──")
    metas = load_issue_ids()
    print(f"  {len(metas)} issues to download")
    out = DATA_DIR / "issues_raw.json"

    # Resume from checkpoint
    if out.exists():
        with open(out) as f: issues = json.load(f)
        done = {i["issue_id"] for i in issues}
        print(f"  Resuming from checkpoint ({len(done)} done)")
    else:
        issues, done = [], set()

    for i, meta in enumerate(metas):
        iid = meta["issue_id"]
        if iid in done: continue
        print(f"  [{i+1}/{len(metas)}] {iid} ...", end=" ", flush=True)
        raw = fetch_issue(iid)
        if raw:
            issues.append(parse_issue(raw, meta))
            print("✓")
        else:
            print("✗")
        if len(issues) % 100 == 0:
            with open(out, "w") as f: json.dump(issues, f)
        time.sleep(0.3)

    with open(out, "w") as f: json.dump(issues, f, indent=2)
    print(f"\n  ✓ Saved {len(issues)} issues → data/issues_raw.json")
    return issues

# =============================================================================
# STEP 2 — Preprocessing
# =============================================================================

def get_wn_pos(tag):
    return {
        "J": wordnet.ADJ, "V": wordnet.VERB,
        "N": wordnet.NOUN, "R": wordnet.ADV
    }.get(tag[0], wordnet.NOUN)

def preprocess(text, lemmatizer):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"`[^`]*`|\{[^}]*\}", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    tokens = [lemmatizer.lemmatize(w, get_wn_pos(t)) for w, t in tagged]
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    tokens = [ONTOLOGY_MAP.get(t, t) for t in tokens]
    return tokens

def step2_preprocess(issues):
    print("\n── STEP 2: Preprocessing text ──")
    lem = WordNetLemmatizer()
    all_tokens, processed = [], []
    for i, issue in enumerate(issues):
        text = f"{issue['summary']} {issue['description']}"
        tokens = preprocess(text, lem)
        all_tokens.append(tokens)
        processed.append({**issue, "tokens": tokens})
        if (i+1) % 500 == 0: print(f"  {i+1}/{len(issues)} ...")

    # Vocabulary
    total_cnt = Counter(t for doc in all_tokens for t in doc)
    doc_cnt   = Counter(t for doc in all_tokens for t in set(doc))
    n = len(all_tokens)
    vocab_df = pd.DataFrame([
        {"token": tok, "total_freq": cnt,
         "doc_freq": doc_cnt[tok],
         "doc_freq_pct": round(doc_cnt[tok]/n, 4)}
        for tok, cnt in total_cnt.most_common()
    ])
    mask = (vocab_df["total_freq"] >= 5) & (vocab_df["doc_freq_pct"] <= 0.95)
    vocab_list = vocab_df[mask]["token"].tolist()

    print(f"  Vocabulary: {len(vocab_df)} raw → {len(vocab_list)} filtered tokens")
    print(f"  Top 15 tokens:")
    print(vocab_df.head(15).to_string(index=False))

    vocab_df.to_csv(DATA_DIR / "vocabulary.csv", index=False)

    # DTM
    vi = {t: i for i, t in enumerate(vocab_list)}
    rows, cols, data = [], [], []
    for di, tokens in enumerate(all_tokens):
        for tok, cnt in Counter(tokens).items():
            if tok in vi:
                rows.append(di); cols.append(vi[tok]); data.append(cnt)
    dtm = sparse.csr_matrix((data, (rows, cols)), shape=(len(all_tokens), len(vocab_list)))
    print(f"  DTM: {dtm.shape}, sparsity {1-dtm.nnz/(dtm.shape[0]*dtm.shape[1]):.2%}")

    sparse.save_npz(str(DATA_DIR / "dtm.npz"), dtm)
    with open(DATA_DIR / "dtm_feature_names.json","w") as f: json.dump(vocab_list, f)
    with open(DATA_DIR / "issues_processed.json","w") as f: json.dump(processed, f, indent=2)
    print("  ✓ Saved all preprocessing outputs")
    return processed, vocab_df, vocab_list, dtm

# =============================================================================
# STEP 3 — Vocabulary plots
# =============================================================================

def step3_plots(vocab_df):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    print("\n── STEP 3: Generating vocabulary plots ──")
    ONTOLOGY = {"COMPONENT","DATA","CONNECTOR","SOLUTION","QUALITY"}
    top30 = vocab_df.head(30)
    colors = ["#E05A5A" if t in ONTOLOGY else "#4A90D9" for t in top30["token"]]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(top30["token"][::-1], top30["total_freq"][::-1], color=colors[::-1], edgecolor="white")
    ax.set_xlabel("Total Frequency", fontsize=12)
    ax.set_title("Top 30 Most Frequent Tokens (after preprocessing)", fontsize=13, fontweight="bold")
    b1 = mpatches.Patch(color="#4A90D9", label="Regular token")
    b2 = mpatches.Patch(color="#E05A5A", label="Ontology class (replaced)")
    ax.legend(handles=[b1, b2])
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "vocab_top30_freq.png", dpi=150)
    plt.close()

    # Document frequency distribution
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(vocab_df["doc_freq_pct"], bins=20, color="#4A90D9", edgecolor="white", alpha=0.8)
    axes[0].axvline(0.95, color="red", linestyle="--", label="Max threshold (95%)")
    axes[0].set_xlabel("Document Frequency (%)"); axes[0].set_ylabel("Number of Tokens")
    axes[0].set_title("Token Document Frequency Distribution"); axes[0].legend()

    sorted_freq = vocab_df["total_freq"].sort_values(ascending=False).values
    cumulative  = np.cumsum(sorted_freq) / sorted_freq.sum() * 100
    axes[1].plot(range(1, len(cumulative)+1), cumulative, color="#4A90D9", linewidth=2)
    axes[1].axhline(80, color="red", linestyle="--", label="80% coverage")
    axes[1].axhline(95, color="orange", linestyle="--", label="95% coverage")
    axes[1].set_xlabel("Number of Unique Tokens"); axes[1].set_ylabel("Cumulative Coverage (%)")
    axes[1].set_title("Token Coverage of Corpus"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.suptitle("Vocabulary Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "vocab_distribution.png", dpi=150)
    plt.close()

    # Ontology class bars
    ont_df = vocab_df[vocab_df["token"].isin(ONTOLOGY)]
    fig, ax = plt.subplots(figsize=(8, 4))
    x, w = range(len(ont_df)), 0.4
    ax.bar([i-w/2 for i in x], ont_df["total_freq"], w, label="Total freq",  color="#4A90D9")
    ax.bar([i+w/2 for i in x], ont_df["doc_freq"],   w, label="Doc freq",    color="#E05A5A", alpha=0.8)
    ax.set_xticks(list(x)); ax.set_xticklabels(ont_df["token"], fontsize=11)
    ax.set_ylabel("Frequency"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Ontology Class Frequencies", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "vocab_ontology_classes.png", dpi=150)
    plt.close()

    print("  ✓ Saved 3 vocabulary plots to plots/")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  WEEK 1 — Data Download and Vocabulary Creation")
    print("=" * 55)

    raw_path = DATA_DIR / "issues_raw.json"
    if raw_path.exists():
        print(f"\nFound existing {raw_path} — skipping download.")
        print("Delete data/issues_raw.json to re-download.")
        with open(raw_path) as f:
            issues = json.load(f)
    else:
        issues = step1_download()

    processed, vocab_df, vocab_list, dtm = step2_preprocess(issues)
    step3_plots(vocab_df)

    print("\n" + "=" * 55)
    print(f"  WEEK 1 COMPLETE")
    print(f"  Issues processed:    {len(processed)}")
    print(f"  Vocabulary size:     {len(vocab_list)} tokens")
    print(f"  DTM shape:           {dtm.shape}")
    print(f"  All outputs in:      data/  and  plots/")
    print("=" * 55)
