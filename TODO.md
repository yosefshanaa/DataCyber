# TODO — *Data Science in Cyber* Final Project

> Checklist for **PRD.md** + **PLAN.md**. Every box maps to an assignment demand and a rubric row.
> Legend: `[ ]` open · `[~]` in progress · `[x]` done · **(R:xx)** = rubric points · **(spec)** = directly required by `haifaUEX (2).pdf`.
> **Deadline: 2026-07-10 23:59.**

---

## Phase 0 — Setup & Source Selection  (R:10 — Problem Understanding & Source Selection)
- [ ] Shortlist 2–3 candidate sources (article/blog/tutorial) in an allowed topic **(spec)**
- [ ] Verify each candidate: clearly defines a problem **(spec)**
- [ ] Verify each: proposes a solution **(spec)**
- [ ] Verify each: has implementation / GitHub repo **(spec)**
- [ ] Verify each: provides data or enough info to reproduce **(spec)**
- [ ] Confirm topic ∈ allowed list (Anomaly/IDS/Malware/Phishing/Fraud/Graph/Privacy/Adversarial/Time-Series/approved) **(spec)**
- [ ] Score candidates on testable-claims + likely-flaws + theory-leverage (PLAN §3)
- [ ] **Lock primary source + pick a backup**
- [ ] Write "why chosen / why others rejected" note (banks R:10)
- [ ] Scaffold repo (PLAN §4): `report/ notebooks/ src/ data/ figures/`
- [ ] Create `requirements.txt` with pinned versions
- [ ] Download/fetch dataset deterministically (`data/get_data.py` or committed)
- [ ] Set global `RANDOM_STATE = 42`
- [ ] **Gate M1:** original repo runs (or repro failure documented)

## Phase 1 — Faithful Reproduction  (R:20 critical / Reproducibility §6.4)
- [ ] Run the author's original code end-to-end **(spec: code executes successfully)**
- [ ] Record exact errors, versions, and fixes applied
- [ ] Confirm all required files & dependencies are available **(spec)**
- [ ] Identify hidden / undocumented preprocessing steps **(spec)**
- [ ] Capture the author's headline results (the numbers we will test)
- [ ] Write overall **reproducibility verdict** (reproducible/partial/no) + evidence **(spec)**
- [ ] **Gate M2:** author pipeline re-run inside our notebook

## Phase 2 — Data Loading & EDA  (R:15 — EDA)
### Data Loading (notebook §1) **(spec)**
- [ ] Data loading
- [ ] Data inspection
- [ ] Report data size & feature types
- [ ] Temporal analysis
- [ ] Missing-value analysis
- [ ] Analyze column & index names — *do they make sense?*
- [ ] Handle single-value / constant / irrelevant features
- [ ] Detect & handle duplicated features
### EDA (notebook §2) **(spec)**
- [ ] Feature distributions
- [ ] Missing-values analysis (EDA view)
- [ ] Outlier analysis — Z-score **and** robust (IQR / MAD / Modified-Z)
- [ ] Temporal features handled — *aligns with world knowledge?* (hours/weekends/holidays/UTC/DST/cyclical)
- [ ] Crosstab / group-by analysis
- [ ] Correlation analysis — **justify Pearson vs Spearman vs Kendall** (math meaning, assumptions, pros, cons)
- [ ] State practical-vs-statistical significance of correlations for the cyber question
- [ ] Class imbalance / prevalence — real-world meaning
- [ ] Is there a sampling problem in a class? **Did the authors address it?**
- [ ] Relevant visualizations (titled axes, stated takeaways)
- [ ] Start the **claims table** (PLAN §6)
- [ ] **Gate M3:** EDA complete; imbalance/temporal/correlation findings logged

## Phase 3 — Feature Engineering & Redundancy  (R:10 — Feature Engineering)
### Notebook §3 **(spec)**
- [ ] Encode categorical variables — **state which method and why**
- [ ] Feature scaling
- [ ] Feature creation (e.g., Δt between events, rolling counts, cyclical time)
- [ ] Feature selection
- [ ] Dimensionality reduction (when appropriate)
- [ ] **All transforms fit on train only** (Pipeline/ColumnTransformer — no leakage)
### Report §6.3 analysis **(spec)**
- [ ] State whether the author did feature engineering; list features used
- [ ] For each transform (log/Box–Cox/standardize/min–max/one-hot/target-enc/aggregation/creation/selection/dim-reduction): **why → problem addressed → effect**, with before/after evidence
- [ ] **Redundancy:** how to spot it (corr/VIF/MI/duplicate/constant) **and** how to tackle it
- [ ] Argue whether FE was meaningful — **math intuition + cybersecurity angle**
- [ ] Propose additional features that could improve performance
- [ ] **Gate M4:** leakage-free features; redundancy handled

## Phase 4 — Modeling, Evaluation, Error Analysis  (R:15 Models + R:10 Eval/Error)
### Model Training (notebook §4) **(spec)**
- [ ] Add a sensible **baseline** (majority-class / simple rule)
- [ ] Train **model #1** (from allowed list)
- [ ] Train **model #2** (from allowed list) — *≥2 models required*
- [ ] Same preprocessing for all models (fair comparison)
- [ ] Fixed seeds; **train/test split or cross-validation** **(spec, code quality)**
### Evaluation (notebook §5) **(spec)**
- [ ] Choose problem-appropriate metrics; **justify selections & exclusions**
- [ ] Classification: Accuracy, Precision, Recall, F1, **Fβ**, **MCC**, ROC-AUC, Confusion Matrix (+ PR-AUC for imbalance)
- [ ] *or* Anomaly: Precision, Recall, F1, Fβ, ROC-AUC (when applicable), MCC
- [ ] *or* Regression: MAE, MSE, RMSE, R²
- [ ] For **each** metric: mathematical definition **+** cybersecurity interpretation **(spec)**
- [ ] Discuss FP vs FN implications in this specific domain **(spec)**
- [ ] Model **comparison table**
- [ ] Threshold tuning showing the FP↔FN trade-off
### Error Analysis (notebook §6) **(spec)**
- [ ] Concrete examples of model failures
- [ ] Patterns in the errors
- [ ] Cybersecurity implications of the errors
- [ ] The False-Positive / False-Negative trade-off
- [ ] **Gate M5:** models + metrics + error analysis complete

## Phase 5 — Critical Evaluation Synthesis  (R:20 — the crown jewel)  **(spec)**
- [ ] List the author's main claims verbatim (with location)
- [ ] For each claim: supported by the presented evidence? (Yes/Partial/No + why)
- [ ] Is the evaluation methodology appropriate? (metric/split/baseline/leakage)
- [ ] Possible weaknesses / limitations
- [ ] Are the conclusions justified?
- [ ] **If our findings contradict the author, explain why** (with the proving figure)
- [ ] Run stress tests: prevalence yardstick, metric swap, leakage probes, robustness, drift, seed/CV stability (PLAN §6)
- [ ] Finalize the **claims table** (claim → evidence → re-test → verdict)
- [ ] **Gate M6:** every claim verdicted with our own evidence

## Phase 6 — Report (PDF, English) & Polish
### Report sections **(spec — all required)**
- [ ] §1 Summary of the Source: problem · why important · solution · dataset · model/methodology (R:15)
- [ ] §2 Critical Evaluation (R:20)
- [ ] §3 Feature Engineering Analysis (R:10)
- [ ] §4 Reproducibility Analysis
- [ ] §5 Experimental Results: experiments · modifications · models · metrics · results
- [ ] §6 Conclusions: key findings · lessons · strengths/weaknesses · future improvements
- [ ] §7 **Executive Summary (~1 page, standalone)**
- [ ] §8 **Summing It Up** — emphasize critique of the *source* (not our code), and include:
  - [ ] problem · selected article · dataset · methodology
  - [ ] main findings of our reproduction study
  - [ ] **whether the author's claims were supported by our results**
  - [ ] most important insights
  - [ ] **recommendation: use this approach on similar problems? (yes/no/caveats)**
  - [ ] final conclusion
- [ ] Export report to **PDF** in English → `report/report.pdf`
- [ ] Verify every report number traces to a notebook cell (G-trace)
### Code Quality (R:5) **(spec)**
- [ ] Short, focused functions
- [ ] Meaningful variable names
- [ ] No unnecessary loops (vectorized pandas/numpy)
- [ ] Proper use of pandas / numpy / scikit-learn
- [ ] Clear separation: preprocessing / EDA / training / evaluation
- [ ] Accurate, informative **English** comments
- [ ] No duplicated code
- [ ] Fixed random seeds
- [ ] Train/test split or cross-validation used
### README **(spec)**
- [ ] Project description
- [ ] Link to the selected article / blog / tutorial
- [ ] Link to the original GitHub repository
- [ ] Execution instructions
- [ ] Dataset source
- [ ] **Gate M7:** PDF + README + clean code done

## Phase 7 — Submit  **(spec)**
- [ ] Fresh-kernel **Restart & Run All** passes
- [ ] `pip install -r requirements.txt` works from a clean environment
- [ ] Repo is **public** and contains: PDF report · notebook · supporting code · README
- [ ] Commit & push everything
- [ ] **Email the GitHub repo link to the examiner**
- [ ] Submitted before **2026-07-10 23:59**
- [ ] **Gate M8:** done

---

## Definition of Done (every deliverable)
- [ ] Maps to a PRD requirement ID + a rubric row
- [ ] Numbers/figures produced by the committed, clean-running notebook
- [ ] Written in English; would read as "complete" to the grader
- [ ] Critical items carry a claim → evidence → verdict chain
- [ ] Committed and pushed to the public repo

## Oral-defense readiness (ungraded but gated by spec)
- [ ] Can explain every methodology choice
- [ ] Can explain every code cell (no unexplained copy-paste)
- [ ] Can defend each metric choice and each claim verdict
- [ ] Keep a short decisions log to rehearse from

---

### Rubric coverage tracker (delivered)
| Component | Pts | Delivered in | Status |
|---|---:|---|---|
| Problem Understanding & Source Selection | 10 | README + report §1 (IDS; NSL-KDD; cited sources) | [x] |
| Summary Quality | 15 | report §1 | [x] |
| Critical Evaluation of Author's Claims | 20 | report §2 (claims table + A/B controlled experiment) | [x] |
| Feature Engineering Analysis | 10 | report §3 + notebook §3 (redundancy, leakage, transforms) | [x] |
| Exploratory Data Analysis | 15 | notebook §2 (robust outliers, Spearman, prevalence, shift) | [x] |
| Model Training & Comparison | 15 | notebook §4 (baseline+3 models, CV, A/B, cost-sensitive) | [x] |
| Evaluation & Error Analysis | 10 | notebook §5–6 (MCC/PR-AUC/per-class, threshold sweep) | [x] |
| Code Quality & SW Engineering | 5 | src/ modules, seeds, Pipelines, English comments | [x] |
| **Total** | **100** | | |

> Headline evidence: RF accuracy 0.9990 (random split) → 0.7798 (official KDDTest+); R2L/U2R recall ≈ 0.05.
