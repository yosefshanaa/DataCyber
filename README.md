# DataCyber — Critical Reproduction Study: ML Intrusion Detection on NSL-KDD

**Course:** Data Science in Cyber — Dr. Uri Itai · **Topic:** Intrusion Detection Systems (IDS)
**Author:** Yosef Shanaa · **ID:** 213314859 · **Repository:** <https://github.com/yosefshanaa/DataCyber>
**Reproducibility:** Python 3.14 · all seeds fixed (`RANDOM_STATE = 42`) · **41/41 automated tests passing**

> **Where to start (for review):** the written deliverable is **[`report/report.pdf`](report/report.pdf)**.
> The end-to-end, executed analysis is **[`notebooks/analysis.ipynb`](notebooks/analysis.ipynb)** (all
> outputs embedded), generated from the plain-text source `notebooks/analysis.py` and the reusable,
> tested modules in `src/`. Every quantitative claim is reproduced into `results/metrics.json`.

---

## TL;DR — what we found
- A widely-copied tutorial reports **~99% accuracy** on NSL-KDD. We show that number reflects the
  **evaluation protocol**, not real detection capability.
- **Controlled A/B experiment** (only the test set changes): the *same* Random Forest scores **0.999**
  on a random split but **0.78** on the official `KDDTest+` — a **21.9-point** drop, **stable to
  ±0.01 pp across five seeds**.
- Under imbalance-aware metrics the supervised models **miss ~95% of R2L/U2R** — the most damaging
  intrusions — even though aggregate accuracy looks high.
- We don't merely recommend a fix, we **test** it: a one-class anomaly detector trained on *normal
  traffic only* **beats every supervised model** on `KDDTest+` (MCC 0.72 vs 0.66) and recovers the rare
  attacks (R2L 0.05→0.47, U2R 0.15→0.78), at a higher false-alarm rate.
- **External validation on the modern UNSW-NB15 (2015)** dataset: the critique reproduces *and sharpens*
  — the gap is smaller on a near-IID split (which confirms the cause), the per-class blind spot
  persists, and anomaly detection proves **conditional** (it wins only when attacks resemble normal
  traffic).

## Project description
This project **critically evaluates** a popular machine-learning tutorial that claims **~99% accuracy**
for network-intrusion detection on the **NSL-KDD** dataset. Rather than merely reproducing the result,
we test whether the claim holds under correct evaluation.

We run a **controlled A/B experiment** that holds the models, features and preprocessing fixed and
changes **only the evaluation protocol**:

- **Protocol A (tutorial-style):** a random 80/20 split of `KDDTrain+` — train and test drawn from the
  *same* distribution.
- **Protocol B (correct):** train on `KDDTrain+`, evaluate on the official, distribution-shifted
  `KDDTest+`.

The headline accuracy reproduces at **~0.999** under Protocol A (and **0.9989 ± 0.0002** under 5-fold
cross-validation) but **collapses to ~0.78 on the official test set** (a 21.9-point drop, **stable to
±0.01 pp over five seeds**), because `KDDTest+` deliberately over-represents the rare, hard attack
families (R2L, U2R). Under honest, imbalance-aware metrics (MCC, Balanced Accuracy, PR-AUC, per-class
recall) the supervised models detect DoS/Probe well but **miss ~95% of the highest-impact intrusions
(R2L/U2R)**. We then **test the implied fix**: a one-class anomaly detector trained on *normal traffic
only* **beats every supervised model on the official test set** (MCC 0.72 vs 0.66) and recovers the rare
attacks (R2L 0.05→0.47, U2R 0.15→0.78), at a higher false-alarm rate. Finally we **replicate the whole
study on the modern UNSW-NB15 (2015) benchmark** to test whether the critique generalises beyond a
single, dated dataset (it does — see §7 of the report).

> **Verdict:** the ~99% claim is *technically computed correctly* but **not supported** as a measure of
> real intrusion-detection capability — it is an artefact of evaluation protocol + metric choice. The
> right paradigm for the dangerous, novel attacks is **anomaly detection** — but, as the UNSW-NB15
> validation shows, only *conditionally*: it helps specifically when the attacks resemble normal traffic.

## Methodology at a glance
- **Design** — a controlled experiment: reproduce the tutorial faithfully, then change *one* variable
  (the evaluation protocol) with models, features, preprocessing and seeds held fixed.
- **EDA** — class balance & train↔test shift, heavy-tail outliers (IQR/MAD), Spearman-vs-Pearson
  correlation & redundancy, PCA separability, protocol×attack-family crosstabs.
- **Feature engineering (leakage-safe)** — unknown-safe one-hot encoding, `log1p` for heavy-tailed
  counts, standardisation, constant-feature drop; implemented as scikit-learn transformers fit on
  *train only*.
- **Models** — majority baseline, Logistic Regression, Random Forest, Gradient Boosting; plus two
  one-class anomaly detectors (Isolation Forest, One-Class SVM).
- **Metrics** — Accuracy, Balanced Accuracy, Precision/Recall, F1, **F2**, **MCC**, ROC-AUC, **PR-AUC**,
  per-class recall and per-family detection rate.
- **Robustness** — 5-fold cross-validation, multi-seed gap, a literal reconstruction of the tutorials'
  own leaky recipe, feature ablations, a decision-threshold sweep, and a second dataset (UNSW-NB15).

**Notebook map (9 sections):** 1 Data Loading & Inspection · 2 EDA · 3 Feature Engineering ·
4 Model Training (the controlled experiment) · 5 Evaluation · 6 Error Analysis ·
7 External Validation (UNSW-NB15) · 8 Executive Summary · 9 Summing It Up.

## Selected source (under review)
- **Tutorial / repository (primary):** *Network Intrusion Detection Using Machine Learning* —
  <https://github.com/abhinav-bhardwaj/Network-Intrusion-Detection-Using-Machine-Learning>
- **Representative of the same pattern (secondary):**
  <https://github.com/Mamcose/NSL-KDD-Network-Intrusion-Detection>
- **Foundational dataset paper:** M. Tavallaee, E. Bagheri, W. Lu, A. Ghorbani, *A Detailed Analysis of
  the KDD CUP 99 Data Set*, IEEE CISDA, 2009.

## Datasets
- **NSL-KDD** (Canadian Institute for Cybersecurity) — *primary study*. Stable mirror:
  <https://github.com/defcom17/NSL_KDD>.
  Files: `KDDTrain+.txt` (125,973 records), `KDDTest+.txt` (22,544 records); 41 features + label +
  difficulty score; attacks grouped into DoS / Probe / R2L / U2R.
- **UNSW-NB15** (Moustafa & Slay, 2015, UNSW Canberra) — *modern external validation* (§7). Official
  partitioned CSVs: `UNSW_NB15_training-set.csv` (175,341 flows), `UNSW_NB15_testing-set.csv` (82,332
  flows); 42 features; nine attack families.

Both datasets are fetched deterministically by `python data/get_data.py` and are also committed under
`data/raw/` for one-clone reproducibility.

## Key results
**1. The headline collapses under correct evaluation.** *Same Random Forest, binary attack/normal —
only the test set changes:*

| Metric | Protocol A (random split) | Protocol B (official KDDTest+) |
|---|---|---|
| Accuracy | 0.9990 | **0.7798** (−21.9 pp) |
| MCC | 0.9981 | 0.6214 |
| Recall (attack) | 0.9985 | 0.6335 |
| R2L recall | — | **0.050** |
| U2R recall | — | **0.045** |
| Majority-baseline accuracy | 0.5346 | 0.4308 (best-possible constant: 0.5692) |

**2. The tested remedy — supervised vs. an anomaly detector trained on normal traffic only
(`KDDTest+`):**

| Model | MCC | F2 | R2L detection | U2R detection | normal FPR |
|---|---|---|---|---|---|
| Random Forest (supervised) | 0.621 | 0.681 | 0.051 | 0.149 | 0.027 |
| Gradient Boosting (supervised) | 0.655 | 0.719 | — | — | — |
| **One-Class SVM (normal-only)** | **0.721** | **0.842** | **0.473** | **0.776** | 0.097 |

The one-class SVM **never sees an attack label** yet outperforms the supervised models on the official
test set and recovers the rare attacks — at the cost of more false alarms.

**3. External validation on modern data (UNSW-NB15).** *Same Random Forest, binary task:*

| Metric | Protocol A (random split) | Protocol B (official test) |
|---|---|---|
| Accuracy | 0.960 | **0.871** (−8.8 pp) |
| MCC | 0.907 | 0.755 |

The A→B gap reproduces but is **milder** (~9 pp vs ~22 pp) because UNSW's official split is near-IID —
confirming that distribution-shift magnitude drives the gap. Aggregate accuracy again hides near-zero
recall on rare families (Analysis ≈ 0.00, Worms ≈ 0.09, Backdoor ≈ 0.10, DoS ≈ 0.10), and here the
anomaly-detection remedy is **conditional**: supervised RF wins (MCC 0.76 vs one-class 0.46/0.17)
because UNSW attacks are flow-separable, whereas NSL-KDD's R2L/U2R mimic normal traffic.

Full numbers are in `report/report.pdf` and `results/metrics.json`; figures are in `figures/`.

## Reproduce it
```bash
# 1. Environment (Python 3.11+; developed and tested on 3.14)
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Fetch the datasets (only needed if data/raw/ is empty)
python data/get_data.py

# 3. Run the full analysis and embed all outputs into the notebook (~15–20 min)
jupytext --to notebook --execute notebooks/analysis.py -o notebooks/analysis.ipynb
#    ...or run it as a plain script (same computation, no embedded notebook):
python notebooks/analysis.py

# 4. (optional) Rebuild the PDF report from report/report.md
python scripts/build_report.py
```
All randomness is fixed via `RANDOM_STATE = 42`, so results are deterministic across runs.

## Tests
A `pytest` suite (41 tests) checks the analysis building blocks: the attack-label taxonomy and its
integrity guard, the leakage-safe preprocessing (including robustness to test-set categories unseen in
training), the imbalance-aware metrics, the model factory, and the UNSW-NB15 loader. Integration tests
run the full pipeline on the bundled data and **lock in the study's central finding — the random-split →
official-test accuracy collapse — as a regression test**.
```bash
pip install -r requirements.txt   # installs pytest
pytest                            # full suite (unit + integration)
pytest -m "not slow"              # fast unit tests only (no dataset required)
pytest -m slow                    # integration tests on the bundled NSL-KDD / UNSW-NB15 data
```

## Assignment coverage
| Required component | Where to find it |
|---|---|
| Source selection & summary | report §1 · README "Selected source" |
| **Critical evaluation of the claims** | report §2, §7 · notebook §4–5, §7 |
| Exploratory data analysis (EDA) | notebook §2 · `src/eda.py` · `figures/` |
| Feature engineering | report §3 · notebook §3 · `src/features.py` |
| Model training & comparison | report §5 · notebook §4–5 · `src/models.py` |
| Evaluation & metric justification | report §5 · notebook §5 · `src/evaluate.py` |
| Error analysis | report §6 · notebook §6 |
| Reproducibility | report §4 · `requirements.txt` · `tests/` · `data/get_data.py` |
| External validation (modern dataset) | report §7 · notebook §7 · `src/unsw.py` |
| Conclusions & recommendations | report §8 · notebook §8–9 |

## Repository layout
```
DataCyber/
├── README.md                  # this file
├── PRD.md / PLAN.md / TODO.md  # planning docs (requirements, plan, checklist)
├── requirements.txt           # pinned, tested dependencies
├── pytest.ini                 # test configuration
├── data/
│   ├── get_data.py            # deterministic fetcher (NSL-KDD + UNSW-NB15)
│   └── raw/                    # NSL-KDD .txt + UNSW-NB15 .csv files
├── src/                       # reusable, leakage-safe analysis modules
│   ├── data.py  eda.py  features.py  models.py  evaluate.py
│   └── unsw.py                # UNSW-NB15 schema/loader (modern-data validation)
├── tests/                     # pytest suite (41 tests)
│   ├── conftest.py            # shared fixtures (synthetic + bundled-data)
│   ├── test_data.py  test_features.py  test_evaluate.py  test_models.py
│   ├── test_unsw.py           # UNSW-NB15 loader/grouping
│   └── test_pipeline_integration.py   # end-to-end, marked "slow"
├── notebooks/
│   ├── analysis.py            # jupytext source of the notebook
│   └── analysis.ipynb         # executed notebook (9 sections, outputs embedded)
├── figures/                   # generated plots used by the report
├── results/metrics.json       # all quantitative results
├── scripts/build_report.py    # report.md -> report.pdf
└── report/
    ├── report.md              # source of the report
    └── report.pdf             # final English report (the graded deliverable)
```

## References
1. M. Tavallaee, E. Bagheri, W. Lu, A. Ghorbani. *A Detailed Analysis of the KDD CUP 99 Data Set.*
   IEEE Symposium on Computational Intelligence for Security and Defense Applications (CISDA), 2009.
2. N. Moustafa, J. Slay. *UNSW-NB15: A Comprehensive Data Set for Network Intrusion Detection Systems.*
   Military Communications and Information Systems Conference (MilCIS), 2015.
3. Reviewed tutorials: `abhinav-bhardwaj/Network-Intrusion-Detection-Using-Machine-Learning`;
   `Mamcose/NSL-KDD-Network-Intrusion-Detection`.

## License / academic integrity
Individual coursework. NSL-KDD © Canadian Institute for Cybersecurity; UNSW-NB15 © Moustafa & Slay,
UNSW Canberra. Reviewed repositories belong to their respective authors and are cited for academic
critique only.
