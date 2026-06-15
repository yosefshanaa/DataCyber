# DataCyber — Critical Reproduction Study: ML Intrusion Detection on NSL-KDD

**Course:** Data Science in Cyber (Dr. Uri Itai) · **Topic:** Intrusion Detection Systems (IDS)

## Project description
This project **critically evaluates** a popular machine-learning tutorial that claims
**~99% accuracy** for network-intrusion detection on the **NSL-KDD** dataset. Rather than merely
reproducing the result, we test whether the claim holds under correct evaluation.

We run a **controlled A/B experiment** that holds the models, features and preprocessing fixed and
changes **only the evaluation protocol**:

- **Protocol A (tutorial-style):** a random 80/20 split of `KDDTrain+` — train and test drawn from
  the *same* distribution.
- **Protocol B (correct):** train on `KDDTrain+`, evaluate on the official, distribution-shifted
  `KDDTest+`.

The headline accuracy reproduces at **~0.999** under Protocol A (and **0.9989 ± 0.0002** under 5-fold
cross-validation) but **collapses to ~0.78 on the official test set** (a 21.9-point drop, **stable to
±0.01 pp over five seeds**), because `KDDTest+` deliberately over-represents the rare, hard attack
families (R2L, U2R). Under honest, imbalance-aware metrics (MCC, Balanced Accuracy, PR-AUC, per-class
recall) the supervised models detect DoS/Probe well but **miss ~95% of the highest-impact intrusions
(R2L/U2R)**. We then **test the implied fix**: a one-class anomaly detector trained on *normal traffic
only* **beats every supervised model on the official test set** (MCC 0.72 vs 0.66) and recovers the
rare attacks (R2L 0.05→0.47, U2R 0.15→0.78), at a higher false-alarm rate. Full numbers are in the
report and `results/metrics.json`.

> **Verdict:** the ~99% claim is *technically computed correctly* but **not supported** as a measure
> of real intrusion-detection capability — it is an artefact of evaluation protocol + metric choice.
> The right paradigm for the dangerous, novel attacks is **anomaly detection**, which we demonstrate,
> not just recommend.

## Selected source (under review)
- **Tutorial / repository (primary):** *Network Intrusion Detection Using Machine Learning* —
  https://github.com/abhinav-bhardwaj/Network-Intrusion-Detection-Using-Machine-Learning
- **Representative of the same pattern (secondary):**
  https://github.com/Mamcose/NSL-KDD-Network-Intrusion-Detection
- **Foundational dataset paper:** M. Tavallaee, E. Bagheri, W. Lu, A. Ghorbani,
  *A Detailed Analysis of the KDD CUP 99 Data Set*, IEEE CISDA, 2009.

## Dataset source
- **NSL-KDD** (Canadian Institute for Cybersecurity). Stable mirror used:
  https://github.com/defcom17/NSL_KDD
- Files: `KDDTrain+.txt` (125,973 records), `KDDTest+.txt` (22,544 records); 41 features + label +
  difficulty score; attacks grouped into DoS / Probe / R2L / U2R.

## Repository layout
```
DataCyber/
├── README.md                 # this file
├── PRD.md / PLAN.md / TODO.md # planning docs (requirements, plan, checklist)
├── requirements.txt          # pinned dependencies
├── data/
│   ├── get_data.py           # deterministic dataset fetcher
│   └── raw/                   # NSL-KDD txt files (committed; ~6 MB)
├── src/                      # reusable, leakage-safe analysis modules
│   ├── data.py  eda.py  features.py  models.py  evaluate.py
├── notebooks/
│   ├── analysis.py           # jupytext source of the notebook
│   └── analysis.ipynb        # executed notebook (8 sections, outputs embedded)
├── figures/                  # generated plots used by the report
├── results/metrics.json      # all quantitative results
├── scripts/build_report.py   # report.md -> report.pdf
└── report/
    ├── report.md             # source of the report
    └── report.pdf            # final English report
```

## Execution instructions
```bash
# 1. Create an environment and install dependencies
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Fetch the dataset (only needed if data/raw/ is empty)
python data/get_data.py

# 3a. Run the analysis as a script (fast sanity run, ~6 min)
python notebooks/analysis.py

# 3b. ...or build & execute the notebook (embeds all outputs)
jupytext --to notebook --execute notebooks/analysis.py -o notebooks/analysis.ipynb

# 4. (optional) Rebuild the PDF report
python scripts/build_report.py
```
All randomness is fixed via `RANDOM_STATE = 42`; results are deterministic.

## Key results (summary)
*Same Random Forest, binary attack/normal — only the test set changes:*

| Metric | Protocol A (random split) | Protocol B (official KDDTest+) |
|---|---|---|
| Accuracy | 0.9990 | **0.7798** (−21.9 pp) |
| MCC | 0.9981 | 0.6214 |
| Recall (attack) | 0.9985 | 0.6335 |
| R2L recall | — | **0.050** |
| U2R recall | — | **0.045** |
| Majority-baseline accuracy | 0.5346 | 0.4308 (best-possible constant: 0.5692) |

*Tested remedy — supervised vs. an anomaly detector trained on normal traffic only (`KDDTest+`):*

| Model | MCC | F2 | R2L detection | U2R detection | normal FPR |
|---|---|---|---|---|---|
| Random Forest (supervised) | 0.621 | 0.681 | 0.051 | 0.149 | 0.027 |
| Gradient Boosting (supervised) | 0.655 | 0.719 | — | — | — |
| **One-Class SVM (normal-only)** | **0.721** | **0.842** | **0.473** | **0.776** | 0.097 |

The one-class SVM **never sees an attack label** yet outperforms the supervised models on the official
test set and recovers the rare attacks — at the cost of more false alarms. See `report/report.pdf` for
the full critical evaluation. Exact figures are generated by the notebook and stored in
`results/metrics.json`.

## License / academic integrity
Individual coursework. Dataset © Canadian Institute for Cybersecurity. Reviewed repositories belong
to their respective authors and are cited for academic critique only.
