# Product Requirements Document (PRD)
### Final Project — *Data Science in Cyber* (Dr. Uri Itai)

> **Source of truth:** `haifaUEX (2).pdf` (assignment spec) and `summary.pdf` (course theory toolkit).
> This PRD is the contract. Every requirement below is traceable to a line in the spec and to a row in the grading rubric. If a deliverable does not satisfy an item here, it is **not done**.

---

## 1. Context & Problem Statement

The "product" is **not software** — it is a **critical reproduction study**: a public GitHub repository containing a **PDF report**, a **Python notebook**, supporting code, and a **README**, which together critically evaluate a *published* article / blog post / tutorial on Data Science in Cybersecurity.

The grader is explicitly **not** rewarding "I re-ran the author's notebook and got the same number." The single largest rubric component (20/100) is **Critical Evaluation of the Author's Claims**. The project succeeds even if it *refutes* the original work — provided the refutation is rigorous, evidenced, and well-communicated.

**One-line product goal:** *Take a real published cyber-DS artifact, reproduce it honestly, prove with evidence which of its claims hold and which do not, and leave the reader able to decide whether to trust the method on similar problems.*

---

## 2. Goals & Non-Goals

### 2.1 Goals (what success looks like)
- G1. Select a **legitimate, reproducible** source that defines a problem, proposes a solution, ships code/GitHub, and provides (or points to) data.
- G2. Faithfully **reproduce** the author's pipeline, documenting every gap, hidden step, and dependency.
- G3. **Critically test** each of the author's claims against evidence we generate ourselves.
- G4. Perform **rigorous EDA and feature-engineering analysis** grounded in *both* mathematical intuition and cybersecurity meaning.
- G5. Train **≥ 2 models**, compare them with **problem-appropriate metrics**, and run **error analysis** centered on the False-Positive / False-Negative trade-off.
- G6. Communicate clearly: report in **English**, clean code, an **Executive Summary** and a **Summing-It-Up** section a busy reader can consume in minutes.
- G7. Ship a **public GitHub repo** meeting all submission requirements, **before 2026-07-10 23:59**.

### 2.2 Non-Goals (explicit scope guards)
- N1. **Not** building a production detector or a novel algorithm. Novelty is optional; rigor is mandatory.
- N2. **Not** critiquing *our own* implementation in the "Summing It Up" section — that section assesses the **original work**.
- N3. **Not** chasing maximum accuracy. A simpler, stable, explainable, well-justified result beats a fragile high score (see `summary.pdf`: *"the best model is not always the most accurate"*).
- N4. **Not** group work. Individual submission; discussion allowed, deliverables must be the student's own.

---

## 3. Stakeholders & Constraints

| Stakeholder | Interest |
|---|---|
| Course staff / examiner | Rubric compliance; evidence of genuine understanding (oral exam possible). |
| Student (owner) | A defensible A-grade submission completed on time. |
| "Reader" persona | Wants to understand problem → method → verdict from the Executive Summary alone. |

**Hard constraints**
- C1. **Language:** report in English. Code comments in English.
- C2. **Deadline:** Friday **2026-07-10 23:59**. Late = rejected without prior approval. (Today is 2026-06-15 → ~3.5 weeks.)
- C3. **Submission medium:** *public* GitHub repository; link emailed to the examiner.
- C4. **Reproducibility:** notebook must be **complete, executable, documented**; fixed seeds; train/test split or CV.
- C5. **Oral defense:** must be able to explain methodology, code, results, conclusions on demand → no copy-paste you cannot justify.
- C6. **Individual integrity:** own work.

---

## 4. Deliverables (the "product")

A single **public GitHub repo** (this repo, `DataCyber`) containing:

| ID | Artifact | Requirement |
|---|---|---|
| D1 | **PDF report** | English; the 6 numbered report sections (§6) + Executive Summary + Summing-It-Up. |
| D2 | **Python notebook** (`.ipynb`) | The 8 notebook areas (§7); runs top-to-bottom without manual fixes. |
| D3 | **Supporting code** | Any `.py` modules, helper scripts, `requirements.txt`/`environment.yml`. |
| D4 | **README.md** | Project description; link to chosen article/blog/tutorial; link to original GitHub repo; execution instructions; dataset source. |
| D5 | **Data access** | Dataset committed if license/size allow, else a script/instructions to fetch it deterministically. |

> **Critical note:** the PDF and the notebook tell the *same story*. Numbers in the report must come from the committed notebook. No orphan figures.

---

## 5. Source Selection Requirements (Rubric: *Problem Understanding & Source Selection*, 10 pts)

The chosen source **must** satisfy ALL of:
- S1. Topic ∈ { Anomaly Detection, Intrusion Detection (IDS), Malware Detection, Phishing Detection, Fraud Detection, Cybersecurity Graph Analytics, Privacy & Data Protection, Adversarial ML, Time-Series for Cybersecurity, *or other topic approved by the lecturer* }.
- S2. **Clearly defines a problem.**
- S3. **Proposes a solution.**
- S4. **Includes an implementation or a GitHub repository.**
- S5. **Provides data, or enough information to reproduce** the experiment.

**Selection quality bar (our addition, to protect the 20-pt critical component):** prefer a source that is *popular enough to have known, testable weaknesses* — e.g., tutorials that report **Accuracy on a highly imbalanced dataset**, scale features **before** the train/test split, train and test on **correlated/duplicated rows**, or claim generalization from a single split. These are exactly the failure modes the course theory arms us to expose (class imbalance, prevalence, MCC vs Accuracy, data leakage, Concept Drift). A "perfect" source gives us little to critique → weaker grade.

**Recommended candidate tracks** (pick one; see PLAN.md §3 for the decision matrix):
- **Fraud Detection** — Kaggle *Credit Card Fraud* (ULB), 0.172% positive class. Classic Accuracy-is-misleading trap.
- **Network IDS** — NSL-KDD / CIC-IDS2017 / UNSW-NB15. Frequent leakage and train/test-from-same-flows issues.
- **Phishing Detection** — UCI / Mendeley phishing URL/website datasets. Many tutorials with leaky or trivially separable features.

> **Anti-requirements (reject a source if):** no runnable code; dead dataset link; proprietary/unavailable data with no substitute; or the "tutorial" is a thin wrapper around `model.fit()` with nothing to evaluate. Document *why* a candidate was rejected — that reasoning itself earns Source-Selection points.

---

## 6. PDF Report Requirements (English)

Each subsection is a checklist item. **Bold = high rubric weight.**

### 6.1 Summary of the Source — (Rubric: *Summary Quality*, 15 pts)
- The problem being addressed.
- Why the problem is important (cyber impact, cost of errors).
- The proposed solution.
- The dataset used (provenance, size, label definition, collection method).
- The model / methodology employed.

### 6.2 Critical Evaluation — (Rubric: *Critical Evaluation*, **20 pts — the crown jewel**)
- Enumerate the author's **main claims** explicitly (quote/cite them).
- For **each** claim: is it **supported by the presented evidence**? (Yes / Partially / No + why.)
- Is the **evaluation methodology appropriate**? (metric choice, split strategy, baseline, leakage, statistical vs practical significance).
- **Weaknesses / limitations** (data, method, scope, threat model).
- Are the **conclusions justified**?
- **If our findings contradict the author, explain why** — with the experiment that shows it.

### 6.3 Feature Engineering Analysis — (Rubric: *Feature Engineering*, 10 pts)
- Was feature engineering performed? Which **features** were used?
- **Each** transformation (log, Box–Cox, standardization, min–max, one-hot, target encoding, aggregation, feature creation, feature selection, dimensionality reduction): **why applied → what problem it addresses → effect on data quality / accuracy / interpretability / predictive performance**, with **evidence** (before/after).
- **Redundancy:** is there redundancy in the system? **How to spot it** (correlation/VIF/duplicate-column/constant-column checks, mutual information) and **how to tackle it** (drop/merge/PCA). *Tie to `summary.pdf`: redundancy harms explainability — importance gets split across correlated features.*
- Was the FE process **meaningful** — explained via **both mathematical intuition and the cybersecurity angle**?
- **Additional features** that could plausibly improve performance (+ justification).

### 6.4 Reproducibility Analysis — (folds into *Critical Evaluation* / *Code Quality*)
- Does the code execute successfully? (exact errors, versions, fixes.)
- Are all **required files and dependencies** available?
- Are there **hidden preprocessing steps** (undocumented filtering, leakage, manual edits)?
- Overall reproducibility verdict (reproducible / partially / not) with evidence.

### 6.5 Experimental Results — (Rubric: *Model Training & Comparison* 15 + *Evaluation & Error Analysis* 10)
- Experiments performed; **modifications introduced**; models trained; metrics; obtained results (tables + plots).

### 6.6 Conclusions
- Key findings; lessons learned; strengths & weaknesses of the proposed solution; concrete future improvements.

### 6.7 Executive Summary
- ≈ **one page**, standalone, plain language.

### 6.8 Summing It Up
- Concise project summary **emphasizing critical evaluation of the source** (not our own code). Must explicitly include: problem; selected article/blog/tutorial; dataset; methodology; **main findings of the reproduction study**; **whether the author's claims were supported by our results**; most important insights; **recommendation (use this approach on similar problems? yes/no/with caveats)**; final conclusion.

---

## 7. Python Notebook Requirements (complete, executable, documented)

### 7.1 Data Loading
- Load data; inspect; report **size & feature types**; **temporal analysis**; **missing-value analysis**; **analyze column & index names — do they make sense?**; handle **single-value / irrelevant features** and **duplicated features**.

### 7.2 Exploratory Data Analysis (EDA) — (Rubric: *EDA*, 15 pts)
- Feature **distributions**; missing-value analysis; **outlier analysis** (Z-score *and* robust: IQR / MAD / Modified-Z — per `summary.pdf`, prefer robust under heavy tails / skew).
- **Temporal features** handled correctly — and *does it align with world knowledge?* (working hours, weekends, holidays, time zones / UTC, DST, cyclical encoding sin/cos).
- **Crosstab / group-by**; **correlation analysis** — **justify** Pearson vs Spearman vs Kendall (math meaning, assumptions, advantages, limitations) and argue **practical vs statistical significance** for the cyber question.
- **Class imbalance / prevalence** — real-world meaning; is there a **sampling problem** in a class; **did the authors address it?**
- Relevant **visualizations** (clear titles, axes, takeaways).

### 7.3 Feature Engineering
- Categorical **encoding (which method + why)**; **scaling**; **feature creation**; **feature selection**; **dimensionality reduction** when appropriate. *All transforms fit on train only (no leakage).*

### 7.4 Model Training — (Rubric: *Model Training & Comparison*, 15 pts)
- Train **≥ 2 models** from the allowed list (LogReg, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, SVM, KNN, Isolation Forest, LOF, Autoencoder). Include a **sensible baseline** (e.g., majority-class / simple rule) to contextualize gains.

### 7.5 Evaluation — (Rubric folded into *Evaluation & Error Analysis*, 10 pts)
- Use **problem-appropriate** metrics, each with **mathematical definition + cyber interpretation**:
  - Classification: Accuracy, Precision, Recall, F1, **Fβ**, **MCC (Matthews Correlation Coefficient)**, ROC-AUC, **Confusion Matrix** (add PR-AUC for imbalance).
  - Anomaly detection: Precision, Recall, F1, Fβ, ROC-AUC (when applicable), MCC.
  - Regression (if applicable): MAE, MSE, RMSE, R².
- **Justify** which metrics were chosen, which were **excluded and why**; discuss **FP vs FN implications** in the specific domain. *A thoughtful, justified metric story is valued over mechanically dumping every metric.*

### 7.6 Error Analysis
- Concrete **failure examples**; **patterns** in errors; **cyber implications**; the **FP/FN trade-off** (and threshold tuning).

### 7.7 Executive Summary (notebook-level wrap)
### 7.8 Summing It Up (mirrors §6.8)

---

## 8. Code Quality Requirements (Rubric: *Code Quality*, 5 pts)
- Short, focused functions; meaningful names; **no unnecessary loops** (vectorize with pandas/numpy); proper use of pandas/numpy/scikit-learn; **clear separation** of preprocessing / EDA / training / evaluation; accurate **English** comments; **no duplicated code**; **fixed random seeds**; **train/test split or cross-validation**.

---

## 9. Submission Requirements
- **Public GitHub** repo with: PDF report, notebook, supporting code, and a **README** containing: project description, link to the article/blog/tutorial, link to the original GitHub repo, execution instructions, dataset source.
- Email the repo link to the examiner.
- **Deadline:** 2026-07-10 23:59.

---

## 10. Acceptance Criteria ↔ Grading Rubric (traceability)

| Rubric component | Pts | Acceptance criteria (must all pass) | PRD refs |
|---|---:|---|---|
| Problem Understanding & Source Selection | 10 | Valid source meeting S1–S5; rejected alternatives documented; problem stated precisely. | §5 |
| Summary Quality | 15 | §6.1 fully covered; dataset & label semantics correct. | §6.1 |
| **Critical Evaluation of Author's Claims** | **20** | Every main claim listed & verdicted with our own evidence; methodology critique incl. leakage/metric/split; contradictions explained. | §6.2, §6.4 |
| Feature Engineering Analysis | 10 | Each transform justified w/ evidence; redundancy detection+remedy; math+cyber rationale; extra-feature proposals. | §6.3, §7.3 |
| Exploratory Data Analysis | 15 | Distributions, outliers (robust), temporal-vs-world-knowledge, correlation choice justified, imbalance/prevalence, visuals. | §7.1, §7.2 |
| Model Training & Comparison | 15 | ≥2 models + baseline; identical preprocessing; fair comparison table; seeds; CV/split. | §7.4 |
| Evaluation & Error Analysis | 10 | Metric definitions+cyber meaning; metric selection justified; FP/FN trade-off; error patterns. | §7.5, §7.6 |
| Code Quality & SW Engineering | 5 | §8 satisfied; notebook runs clean end-to-end. | §8 |
| **Total** | **100** | All deliverables D1–D5 present; submitted on time. | §4, §9 |

---

## 11. Critical-Thinking Charter (how we earn the 20 points)

For **every** author claim, run this loop and record it in a claims table:

1. **State** the claim verbatim (with location).
2. **Identify** what evidence *would* support it.
3. **Check** whether the author provided that evidence.
4. **Reproduce / re-test** it ourselves (re-split, re-metric, re-seed).
5. **Stress-test** with the course toolkit:
   - *Metric honesty:* Accuracy on imbalanced data? Re-report with **MCC / PR-AUC / Fβ**.
   - *Leakage:* scaling/encoding/feature-selection fit before split? duplicated or near-duplicate rows across split? target leakage from a feature?
   - *Robustness:* heavy tails / skew → Z-score over-flags; switch to **IQR/MAD/Modified-Z**; consider **log / Box–Cox**.
   - *Temporal validity:* random split on time-ordered data? test for **Concept Drift** (KS / PSI / Wasserstein between periods).
   - *Generalization:* single split vs **cross-validation**; variance across seeds.
   - *Prevalence:* does the base rate make the "good" score meaningless? (a trivial constant predictor's score as a yardstick).
6. **Verdict:** Supported / Partially / Refuted — with the figure or table that proves it.

> Guiding quotes from `summary.pdf` to weave through the report: *"A good model that doesn't help defend the org is not good"*, *"False Positives exhaust the org; False Negatives endanger it"*, *"All models are wrong, but some are useful."*

---

## 12. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Source can't be reproduced (dead data / broken code) | Med | High | Vet 2–3 sources in week 1; keep a backup source; document repro failures as findings (still scores). |
| Data too large / licensed | Med | Med | Use a documented subset/seed; fetch script; cite source instead of committing raw data. |
| Time-ordered data split randomly (leakage) | High | High | Use temporal split where appropriate; report both; discuss drift. |
| Over-engineering / scope creep | Med | Med | Lock scope to PRD; "≥2 models + baseline," not 8 models. |
| Metric theater (dumping all metrics) | Med | Low | Justify each metric; explicitly exclude the uninformative ones. |
| Notebook won't run clean for grader | Med | High | "Restart & Run All" gate before each commit; pin versions in `requirements.txt`. |
| Oral-exam gap (can't explain a step) | Low | High | Author every cell; no unexplained copy-paste; keep a decisions log. |
| Deadline slip | Low | High | Milestone schedule in PLAN.md with a 4-day buffer before 07-10. |

---

## 13. Definition of Done (DoD)
A deliverable is **done** only when:
- [ ] It maps to a PRD requirement ID and a rubric row.
- [ ] Its numbers/figures are produced by the committed, clean-running notebook.
- [ ] It is written in English and survives a "would the grader mark this complete?" read.
- [ ] For critical-evaluation items: a claim → evidence → verdict chain exists.
- [ ] It is committed and pushed to the public repo.

---

## 14. Glossary / Theory Hooks (from `summary.pdf`)
EDA goals · Pearson/Spearman/Kendall · CrossTab & alias detection · uni/multivariate anomalies · Z-score / IQR / **MAD / Modified-Z** · Isolation Forest · LOF · Explainability & redundancy · Log/Box–Cox/Yeo-Johnson · Skewness (Pearson 1st/2nd, Bowley) · Class imbalance, prevalence, **Laplace smoothing** · Distance metrics: **Wasserstein/EMD, KL, JS, KS, TV, Hellinger** (drift detection) · Graph analytics: modularity, Louvain, Laplacian, spectral clustering, min-cut/max-flow, centralities, friendship paradox · Privacy: K-anonymity, differential privacy, identifiers/quasi-identifiers · Adversarial ML, model theft/inversion/membership inference · Time features: UTC/DST/cyclical, Unix epoch, Δt · **Concept Drift**, SolarWinds (APT/supply-chain/lateral movement) · Metrics: Accuracy/Precision/Recall/F1/Fβ/MCC/ROC-AUC · `argmin Σ(x−a)² = mean`, `argmin Σ|x−a| = median`.
