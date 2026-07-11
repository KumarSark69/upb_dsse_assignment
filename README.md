# Assignment 3 — Topic Modeling of Software Design Issues

## Overview
This assignment explores the topics discussed in software design issues using topic modeling algorithms (LDA and BERTopic), and analyzes their characteristics and co-occurrences with design decision types.

---

## Folder Structure

Place all three scripts and `Issues.xlsx` in the same folder. All scripts share the same `data/`, `plots/`, and `results/` directories.

```
assignment/
├── week1_complete.py       ← run first
├── week2_complete.py       ← run second
├── week3_complete.py       ← run third
├── Issues.xlsx
├── data/
│   ├── issues_raw.json             (week1)
│   ├── issues_processed.json       (week1)
│   ├── vocabulary.csv              (week1)
│   ├── dtm.npz                     (week1)
│   ├── dtm_feature_names.json      (week1)
│   ├── coherence_scores.json       (week2)
│   ├── lda_results.json            (week2)
│   ├── issues_with_topics.json     (week2)
│   └── bertopic_results.json       (week2, if BERTopic installed)
├── plots/
│   ├── vocab_top30_freq.png        (week1)
│   ├── vocab_distribution.png      (week1)
│   ├── vocab_ontology_classes.png  (week1)
│   ├── lda_coherence.png           (week2)
│   ├── lda_top_words.png           (week2)
│   ├── lda_issues_per_topic.png    (week2)
│   ├── lda_topic_proportions.png   (week2)
│   ├── lda_doc_topic_heatmap.png   (week2)
│   ├── bertopic_top_words.png      (week2, if BERTopic installed)
│   ├── bertopic_issues_per_topic.png (week2, if BERTopic installed)
│   ├── rq2_boxplots_lda.png        (week3)
│   ├── rq2_boxplots_bertopic.png   (week3, if BERTopic ran)
│   ├── rq3_lda_vs_bert.png         (week3, if BERTopic ran)
│   ├── rq3_lda_vs_dd1.png          (week3)
│   ├── rq3_lda_vs_dd2.png          (week3)
│   └── rq3_lda_vs_dd3.png          (week3)
└── results/
    ├── rq1_topics.csv              (week2)
    ├── rq2_stats.csv               (week3)
    └── rq3_stats.csv               (week3)
```

---

## Setup (run once)

```bash
pip install requests pandas openpyxl nltk scipy scikit-learn gensim \
            matplotlib seaborn statsmodels bertopic sentence-transformers
```

> `bertopic` and `sentence-transformers` are optional. If not installed, Week 2 and Week 3 run using LDA only and skip BERTopic gracefully.

---

## How to Run

```bash
python week1_complete.py
python week2_complete.py
python week3_complete.py
```

Each script reads from the outputs of the previous one. If a required file is missing it prints a clear error telling you which script to run first.

---

## Week 1 — Data Download and Vocabulary Creation

**Goal:** Prepare issue data and build the vocabulary for topic modeling.

**What we did:**
1. **Downloaded and filtered issue data** — Fetched all issues from the Apache Jira API across 5 projects, collecting key fields and excluding bot comments from 35 known bot authors.
2. **Preprocessed issue text** — Concatenated summary and description per issue, then applied tokenization, stop word removal, lemmatization, and ontology class replacement (COMPONENT, CONNECTOR, DATA, SOLUTION, QUALITY).
3. **Built vocabulary and Document-Term Matrix** — Created a filtered vocabulary from all unique tokens and constructed a sparse DTM ready as input for LDA topic modeling in Week 2.

**Output files:** `data/issues_raw.json`, `data/issues_processed.json`, `data/vocabulary.csv`, `data/dtm.npz`, `data/dtm_feature_names.json`

---

## Week 2 — Topic Modeling (LDA + BERTopic)

**Goal:** Discover topics in issues and answer RQ1.

**What we did:**
1. **Ran coherence scoring** — Tested LDA with 3 to 10 topics using α = β = 0.01 and selected the number of topics with the highest C_V coherence score.
2. **Trained final LDA and ran BERTopic** — Trained the final LDA model with the optimal number of topics, computed topic proportions per issue, and ran BERTopic with sentence embeddings for an independent topic discovery.
3. **Answered RQ1** — Produced a topic table with the top keywords per topic and the number of issues assigned to each topic for both LDA and BERTopic.

**Output files:** `data/lda_results.json`, `data/issues_with_topics.json`, `data/bertopic_results.json`, `results/rq1_topics.csv`

---

## Week 3 — Statistical Analysis of Topics (RQ2 + RQ3)

**Goal:** Analyze topic characteristics and co-occurrences to answer RQ2 and RQ3.

**What we did:**
1. **Analyzed issue characteristics per topic (RQ2)** — Generated box plots for four characteristics (comments, human comments, attachments, description length) per LDA and BERTopic topic, and applied Kruskal-Wallis and pairwise Mann-Whitney U tests with Bonferroni correction to identify significant differences.
2. **Conducted co-occurrence significance tests (RQ3)** — Built contingency tables and applied chi-square tests to assess whether LDA topics and BERTopic topics significantly co-occur with each other and with the three design decision types (dd1, dd2, dd3).
3. **Saved all results** — Exported statistical results to `rq2_stats.csv` and `rq3_stats.csv`, and saved co-occurrence heatmaps and box plots to the `plots/` folder for reporting.

**Output files:** `results/rq2_stats.csv`, `results/rq3_stats.csv`, `plots/rq2_*.png`, `plots/rq3_*.png`

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Jira API gives 403 or timeout | Increase `time.sleep(0.3)` to `1.0` in week1; run from your own machine |
| BERTopic not installed | `pip install bertopic sentence-transformers`; scripts skip gracefully |
| `issues_processed.json` not found | Run `week1_complete.py` first |
| `issues_with_topics.json` not found | Run `week2_complete.py` first |
| Poor LDA topics | Add more terms to `STOP_WORDS` or `ONTOLOGY_MAP` in week1 and re-run |
| Low coherence scores | Try wider n range or lower `min_freq` in week1 vocabulary step |
