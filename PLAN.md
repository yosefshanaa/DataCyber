# Implementation Plan
### Final Project — *Data Science in Cyber* (Dr. Uri Itai)

> Companion to **PRD.md** (the *what*) and **TODO.md** (the *checklist*). This is the *how* and *when*.
> **Today:** 2026-06-15 · **Deadline:** 2026-07-10 23:59 · **Working window:** ~25 days.

---

## 1. Strategy in one paragraph

We treat the project as a **prosecution, not a tribute**. We pick a popular, reproducible cyber-DS tutorial whose claims are *checkable*, reproduce it honestly, then use the course toolkit (robust statistics, imbalance/prevalence, leakage detection, drift metrics, proper evaluation) to **test each claim and report the verdict with our own evidence**. The deliverable is a public GitHub repo (PDF + notebook + README) that a busy reader can trust. We optimize for **rigor and clarity over accuracy**, and we protect the schedule with a hard "Restart & Run All" gate and a 4-day buffer.

---

## 2. Phases & Milestones (timeline back from 2026-07-10)

| Phase | Dates (2026) | Milestone / Exit gate |
|---|---|---|
| **P0 — Setup & source selection** | Jun 15 – Jun 19 | **M1:** Source locked (meets S1–S5), backup chosen, data fetches, original repo runs (or documented failure). Repo scaffolded. |
| **P1 — Faithful reproduction** | Jun 19 – Jun 23 | **M2:** Author pipeline re-run end-to-end in *our* notebook; repro gaps & hidden steps logged. |
| **P2 — Data understanding & EDA** | Jun 23 – Jun 27 | **M3:** §7.1–7.2 complete; claims table drafted; imbalance/temporal/correlation findings in hand. |
| **P3 — Feature engineering & redundancy** | Jun 27 – Jun 30 | **M4:** §7.3 complete, leakage-free; redundancy detected+remedied; extra-feature proposals. |
| **P4 — Modeling & evaluation** | Jun 30 – Jul 4 | **M5:** ≥2 models + baseline; metric suite; error analysis; comparison table. |
| **P5 — Critical evaluation synthesis** | Jul 4 – Jul 6 | **M6:** Every claim verdicted with evidence; contradictions explained. |
| **P6 — Report & polish** | Jul 6 – Jul 8 | **M7:** PDF report (all 8 sections), README, code cleanup, clean run verified. |
| **P7 — Buffer / submit** | Jul 8 – Jul 10 | **M8:** Final "Restart & Run All", push, **email link to examiner**. |

> **Buffer rationale:** two full days (Jul 8–10) absorb the classic last-minute failure (notebook won't run clean on a fresh kernel). Do **not** plan work into the buffer.

---

## 3. P0 — Source Selection (decision matrix)

Vet **2–3 candidates**; score each before committing. A source must pass S1–S5 (PRD §5). Then prefer the one with the **richest, most testable weaknesses**.

| Criterion | Weight | What we look for |
|---|---|---|
| Reproducible code + live data | ★★★ | Repo runs; dataset downloadable deterministically. |
| Testable claims | ★★★ | Specific accuracy/AUC numbers we can re-check. |
| Known/likely methodological flaws | ★★★ | Accuracy on imbalanced data; scale-before-split; duplicated rows; random split on temporal data; no baseline. |
| Theory leverage | ★★ | Lets us deploy MCC/PR-AUC, MAD/IQR, drift metrics, prevalence, Laplace. |
| Scope sanity | ★★ | Finishable in ~3 weeks on a laptop (no 50 GB / GPU-week training). |

**Candidate tracks (pick ONE):**
- **A. Fraud Detection — Kaggle Credit Card Fraud (ULB).** 284k tx, **0.172%** fraud. Textbook Accuracy trap; perfect for MCC/PR-AUC/Fβ, threshold tuning, FP/FN economics. Tons of leaky tutorials. *Lowest risk, highest critique yield.*
- **B. Network IDS — NSL-KDD / CIC-IDS2017 / UNSW-NB15.** Rich features; frequent leakage (flows split randomly), label noise, temporal drift. Strong cyber framing.
- **C. Phishing Detection — UCI / Mendeley phishing.** URL/site features; often trivially separable or leaky; good for feature-engineering + redundancy critique.

**Default recommendation:** **Track A (Credit Card Fraud)** unless the student prefers a network-security flavor (then Track B). Track A maximizes the 20-pt critical-evaluation yield with the least reproducibility risk.

**Output of P0:** `README` stub with links (source + original repo + dataset), `requirements.txt`, repo scaffold, and a short **"why this source / why we rejected the others"** note (banks Source-Selection points).

---

## 4. Repository Layout (target)

```
DataCyber/
├── README.md                # description, links (article + orig repo), how-to-run, dataset source
├── PRD.md / PLAN.md / TODO.md
├── requirements.txt         # pinned versions
├── report/
│   └── report.pdf           # the English PDF (8 sections)
├── notebooks/
│   └── analysis.ipynb       # complete, runs top-to-bottom, fixed seeds
├── src/                     # reusable, tested helpers (no duplicated code)
│   ├── data.py              # load / clean / split (split-aware, no leakage)
│   ├── eda.py               # distributions, robust outliers, correlation, drift
│   ├── features.py          # encoders/scalers/creators (fit on train only)
│   ├── models.py            # model factory + CV
│   └── evaluate.py          # metrics (+MCC, Fβ, PR-AUC), confusion, error analysis
├── data/                    # committed if small/licensed; else fetch script + .gitignore
│   └── get_data.py
└── figures/                 # exported plots used in the report
```

> **Leakage firewall (architectural rule):** every `fit` (scaler, encoder, imputer, selector, resampler like SMOTE) happens **inside** a scikit-learn `Pipeline`/`ColumnTransformer` fit **only on training folds**. This single decision prevents the most common — and most embarrassing — author flaw, and lets us *demonstrate* it when the author got it wrong.

---

## 5. Work breakdown per phase (objectives → activities → outputs)

### P1 — Faithful reproduction (M2)
- **Obj:** establish a truthful baseline = "what the author actually gets."
- **Do:** run original repo as-is; pin versions; log every error/fix; capture the author's headline numbers; note undocumented/manual/hidden steps.
- **Out:** Reproducibility section draft (PRD §6.4); a re-run baseline number; repro-gap log.

### P2 — Data understanding & EDA (M3)
- **Obj:** know the data better than the author did.
- **Do:** shapes, dtypes, **column/index sanity** ("does this make sense?"), missing values, **single-value/constant & duplicated features**, distributions, **robust outliers (IQR/MAD/Modified-Z)** alongside Z-score, **temporal analysis vs world knowledge** (working hours/weekends/holidays/UTC/DST/cyclical), **crosstab/group-by**, **correlation with justified Pearson/Spearman/Kendall choice**, **class imbalance & prevalence** (+ did authors address it?). Start the **claims table**.
- **Out:** §7.1–7.2 cells + figures; EDA findings; imbalance verdict.

### P3 — Feature engineering & redundancy (M4)
- **Obj:** leakage-free, justified features; expose redundancy.
- **Do:** encoding (method + why), scaling, **feature creation** (e.g., Δt between events, rolling counts, cyclical time), **feature selection**, **dimensionality reduction** when warranted; **redundancy detection** (correlation matrix, VIF, mutual information, duplicate/constant columns) + **remedy**; tie each transform to **math intuition + cyber meaning**; list **additional features** that could help.
- **Out:** §7.3 + §6.3; redundancy subsection with before/after evidence.

### P4 — Modeling & evaluation (M5)
- **Obj:** fair, reproducible model comparison with honest metrics.
- **Do:** **baseline** (majority/simple rule) + **≥2 models** (e.g., LogReg + Gradient Boosting/RandomForest, or IsolationForest + Autoencoder for the anomaly framing); identical preprocessing via Pipeline; **fixed seeds**; **CV or temporal split**; full metric suite with **definitions + cyber interpretation**; **PR-AUC/MCC/Fβ** foregrounded for imbalance; **confusion matrices**; **threshold tuning** showing the FP↔FN trade-off; **error analysis** (failure cases, patterns, cyber implications).
- **Out:** §7.4–7.6 + §6.5; comparison table; error-analysis subsection.

### P5 — Critical evaluation synthesis (M6)
- **Obj:** convert evidence into verdicts (the 20-pt core).
- **Do:** finalize the **claims table** (claim → expected evidence → author's evidence → our re-test → verdict); write the methodology critique (leakage, metric honesty, split, baseline, drift, generalization); **explain every contradiction** with the figure that proves it.
- **Out:** §6.2 complete; inputs to §6.6 and §6.8.

### P6 — Report & polish (M7)
- **Obj:** publication-quality, English, self-consistent.
- **Do:** assemble PDF (§6.1–6.8 incl. **Executive Summary ~1 page** and **Summing-It-Up**); ensure every number traces to the notebook; **code cleanup** to §8 (functions, names, no dup, no needless loops, English comments, seeds, split/CV); finalize README (all required links + run instructions + dataset source).
- **Out:** `report/report.pdf`, clean `src/`, README.

### P7 — Buffer / submit (M8)
- **Do:** fresh-kernel **Restart & Run All**; verify `pip install -r requirements.txt` from clean env; final commit & push to **public** repo; **email link to examiner**; tag release.

---

## 6. Critical-Evaluation Methodology (operational)

The **claims table** is the spine of the report. Template:

| # | Author claim (verbatim + loc) | Evidence they gave | Our re-test | Result | Verdict |
|---|---|---|---|---|---|
| 1 | "Model achieves 99.9% accuracy" | single random split | re-report MCC/PR-AUC; constant-predictor baseline | baseline≈99.8% acc, MCC≈0 | **Refuted (misleading metric)** |
| 2 | … | … | … | … | … |

**Standard stress tests to run (course toolkit):**
- **Prevalence yardstick:** compute the trivial constant-predictor score; if it rivals the model's headline metric → the metric is theater.
- **Metric swap:** replace Accuracy with **MCC + PR-AUC + Fβ** (β chosen by FP/FN cost in the domain).
- **Leakage probes:** (a) any transform fit before split? (b) duplicate/near-duplicate rows across split? (c) a feature that encodes the label? (d) time-ordered data split randomly?
- **Robustness:** skew/heavy tails → Z-score over-flags; compare **IQR/MAD/Modified-Z**; try **log/Box–Cox**.
- **Drift:** if temporal, compare period-to-period distributions with **KS / PSI / Wasserstein**; relate to **Concept Drift**.
- **Stability:** vary seed / use CV; report metric variance, not a single lucky split.

---

## 7. Environment & tooling
- Python 3.11; `pandas`, `numpy`, `scikit-learn`, `matplotlib`/`seaborn`, `scipy`; optional `xgboost`/`lightgbm`/`imbalanced-learn`.
- **Pin versions** in `requirements.txt` (reproducibility = graded).
- **Seeds:** one `RANDOM_STATE = 42` constant threaded through splits/models.
- Notebook discipline: **idempotent cells**, run order = top-to-bottom, no reliance on out-of-order execution, no manual file edits mid-run.

---

## 8. Quality gates (must pass to advance a phase)
- **G-run:** notebook executes clean from a fresh kernel.
- **G-leak:** no transform fits on test data anywhere (Pipeline-enforced).
- **G-trace:** every report number is reproducible from a notebook cell.
- **G-critic:** each phase adds ≥1 row of evidence to the claims table.
- **G-english:** all prose/comments in English.

---

## 9. Decisions still open (resolve in P0)
1. **Source/topic** — recommended **Track A (Credit Card Fraud)**; confirm or switch to B/C. *(Materially shapes dataset, models, metrics — settle first.)*
2. **Problem framing** — supervised classification vs unsupervised anomaly detection (changes model list & metric set).
3. **Data handling** — commit subset vs fetch script (license/size dependent).

> When in doubt, default to the choice that **maximizes critical-evaluation yield** and **minimizes reproducibility risk** — that is where the points are.
