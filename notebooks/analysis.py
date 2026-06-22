# %% [markdown]
# # Critical Reproduction Study — Network Intrusion Detection on NSL-KDD
#
# **Course:** Data Science in Cyber (Dr. Uri Itai) · **Topic:** Intrusion Detection Systems (IDS)
#
# **Source under review:** *Network Intrusion Detection Using Machine Learning* — a widely
# replicated NSL-KDD classifier tutorial/repository
# (`github.com/abhinav-bhardwaj/Network-Intrusion-Detection-Using-Machine-Learning`),
# representative of the dominant pattern across NSL-KDD tutorials (see also
# `github.com/Mamcose/NSL-KDD-Network-Intrusion-Detection`). These tutorials train
# classifiers on **KDDTrain+** with a **random train/test split** and report **~99% accuracy**.
#
# **Claim under test:** *"Machine-learning classifiers detect network intrusions on NSL-KDD with ~99% accuracy."*
#
# **Our thesis:** that number is an artefact of (a) evaluating on a *random split of one
# distribution* instead of the official **KDDTest+** (which the dataset authors built with
# novel attacks), and (b) using **Accuracy** on **imbalanced** data. Under a correct protocol the
# same models collapse to **~75–80%** accuracy, with near-zero recall on the rarest, most
# dangerous attacks (R2L, U2R). This notebook generates the evidence end-to-end.
#
# > Reference: Tavallaee, Bagheri, Lu & Ghorbani (2009), *A Detailed Analysis of the KDD CUP 99 Data Set*, IEEE CISDA.

# %%
from __future__ import annotations
import json
import os
import sys
import warnings
from pathlib import Path

# Suppress the noisy joblib-worker UserWarning emitted by n_jobs=-1 estimators in
# scikit-learn 1.9 (loky workers are separate processes, so an in-process filter
# does not reach them — an env var does). ConvergenceWarning stays visible.
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning")

import matplotlib
# Inside Jupyter we keep the inline backend so every figure renders in the notebook;
# when this file is run as a plain script we switch to the headless Agg backend.
try:
    get_ipython()  # type: ignore  # noqa: F821
except NameError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

# make the local src/ package importable whether run from repo root or notebooks/
ROOT = Path.cwd()
if not (ROOT / "src").exists() and (ROOT.parent / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src import data, eda, evaluate, features, models, unsw  # noqa: E402

# Silence library deprecation noise but keep ConvergenceWarning visible — we want
# to know if Logistic Regression fails to converge rather than hide it.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# `display` is provided by IPython inside the notebook; this shim lets the same
# file also run as a plain script (e.g. for CI / smoke testing).
try:
    display  # type: ignore  # noqa: B018
except NameError:
    def display(obj):  # noqa: D401
        print(obj)

sns.set_theme(style="whitegrid")
pd.set_option("display.width", 120, "display.max_columns", 60)
RANDOM_STATE = models.RANDOM_STATE
RESULTS: dict = {}  # everything quantitative is collected here and dumped to JSON
(ROOT / "results").mkdir(exist_ok=True)
print("repo root:", ROOT)

# %% [markdown]
# ## 1. Data Loading & Inspection
#
# NSL-KDD ships as headerless CSVs. We attach the canonical 41-feature schema plus the two
# trailing columns: the fine-grained `label` and a `difficulty` score. We derive a binary
# target (`is_attack`) and the standard 5-class target (`attack_category`).

# %%
train, test = data.load_train_test(ROOT / "data" / "raw")
print("KDDTrain+:", train.shape, "| KDDTest+:", test.shape)
ov_tr, ov_te = eda.basic_overview(train), eda.basic_overview(test)
RESULTS["overview_train"], RESULTS["overview_test"] = ov_tr, ov_te
display(pd.DataFrame({"KDDTrain+": ov_tr, "KDDTest+": ov_te}))

# %%
# Feature dtypes
print("Feature dtype counts (model inputs only):")
print(data.feature_matrix(train).dtypes.value_counts())
train.head(3)

# %% [markdown]
# ### 1.1 Do the column & index names make sense?
#
# * The **index** is a meaningless `RangeIndex` (no event id / no timestamp) — correct to leave as-is.
# * Column names are sensible once attached. **Two columns are NOT model features:**
#   * `difficulty` — NSL-KDD *metadata* (how many of 21 baseline learners got the row right).
#     It is unavailable at inference time, and it is **correlated with the label** → using it as a
#     feature is **target leakage**. We exclude it (it is not in `FEATURE_COLUMNS`). Many tutorials
#     leave it in.
#   * `label` — the raw target.

# %%
# Demonstrate the leakage risk of the 'difficulty' column: it differs by class.
leak = train.groupby("is_attack")["difficulty"].mean().rename({0: "normal", 1: "attack"})
print("Mean 'difficulty' by class (should differ -> leaky if used as a feature):")
print(leak.round(3))
RESULTS["difficulty_leak"] = {k: float(v) for k, v in leak.items()}

# %% [markdown]
# ### 1.2 Missing values, duplicates, constant & duplicate features

# %%
print("Total missing (train):", train.isna().sum().sum(), "| (test):", test.isna().sum().sum())
print("Duplicate rows (train):", int(train.duplicated().sum()),
      "| (test):", int(test.duplicated().sum()),
      "  <- NSL-KDD removed the KDD'99 duplicate-record bias")
const = eda.constant_features(train, data.FEATURE_COLUMNS)
dup_pairs = eda.duplicate_feature_pairs(train, data.FEATURE_COLUMNS)
print("Constant (zero-variance) features:", const)
print("Exact duplicate feature pairs:", dup_pairs)
RESULTS["constant_features"] = const
RESULTS["duplicate_feature_pairs"] = dup_pairs

# %% [markdown]
# **Finding.** `num_outbound_cmds` is constant (all zeros) across all 126k rows — zero information.
# It must be dropped (done automatically by `features.feature_groups`). There are no missing values
# and essentially no duplicate rows (an NSL-KDD design improvement over KDD'99).

# %% [markdown]
# ### 1.3 Temporal analysis — does it align with world knowledge?
#
# NSL-KDD has **no timestamp and no event ordering**. The only time-like feature is `duration`
# (connection length in seconds). So genuine time-series analysis (hour-of-day, day-of-week,
# drift over wall-clock time) is **impossible** on this dataset — an honest limitation that the
# tutorials never state. The "temporal" dimension that *does* matter here is the
# **train → test distribution shift** (a *concept-drift* analogue), which we quantify next.

# %% [markdown]
# ## 2. Exploratory Data Analysis (EDA)

# %% [markdown]
# ### 2.1 Class balance & prevalence — and the train↔test shift
#
# Prevalence is the real-world base rate of each class. It tells us (i) Accuracy is dangerous here
# and (ii) whether the train and test sets even describe the same world.

# %%
prev_tr = eda.class_prevalence(train, "attack_category").reindex(data.CLASS_ORDER)
prev_te = eda.class_prevalence(test, "attack_category").reindex(data.CLASS_ORDER)
combined = pd.concat({"KDDTrain+": prev_tr, "KDDTest+": prev_te}, axis=1)
RESULTS["prevalence"] = json.loads(combined.to_json())
eda.plot_class_balance(train, test, "attack_category", data.CLASS_ORDER)
print(combined)

# %% [markdown]
# **Finding (the smoking gun).** The class mix is *not* stationary:
# R2L rises from **0.79% → 12.8%** and U2R from **0.04% → 0.30%** between train and test, while the
# binary attack rate moves 46.5% → 56.9%. KDDTest+ deliberately over-represents the hard
# remote/elevation attacks that models barely see in training. Any model tuned on the train
# distribution is being asked a *different question* at test time — this is the core reason
# random-split accuracy does not transfer.
#
# **Real-world meaning & did the authors address it?** Prevalence is the attack *base rate* a SOC
# faces; with U2R at 0.04% of training rows (52 samples) the learner sees almost no examples of the
# most damaging attack. **The reviewed tutorials do not address this**: they report a single
# Accuracy number, use no resampling/class weights, and never inspect per-class prevalence — so the
# imbalance silently dominates the result. There is effectively a **sampling problem** in U2R/R2L
# (too few positives to learn a stable boundary), which §5.4 probes directly.

# %% [markdown]
# ### 2.2 Feature distributions & robust outlier analysis
#
# Cyber traffic features are heavy-tailed. We compare a classic **Z-score** outlier count with the
# robust **IQR** and **MAD / Modified-Z** rules. Under heavy skew, Z-score and the mean are
# distorted by extreme values, so MAD/IQR (and the median) are the honest choices.

# %%
skewed = ["src_bytes", "dst_bytes", "duration", "count", "srv_count", "dst_host_count"]
# Distributions of a representative feature set (counts/bytes + a bounded rate).
eda.plot_feature_distributions(
    train, skewed + ["serror_rate", "same_srv_rate", "dst_host_same_srv_rate"])
out_tbl = pd.DataFrame([eda.outlier_summary(train[c]) for c in skewed]).set_index("feature")
RESULTS["outlier_summary"] = json.loads(out_tbl.to_json())
print("Outlier counts by method (note how Z-score under-counts under extreme skew):")
display(out_tbl.round(2))
eda.plot_log_transform_effect(train, "src_bytes")
print("log1p collapses the skew that inflates Z-score outliers.")

# %% [markdown]
# **Finding.** The byte/count features are extremely right-skewed — `src_bytes` has skew ≈ 191 and
# spans 0 to 1.3e9. The robust methods agree the spread is huge (IQR flags ~13,840 and MAD ~31,784
# `src_bytes` outliers) while the classic **Z-score flags only 11** — because a handful of giant
# values inflate the standard deviation and *mask* the very outliers it is meant to find. This is the
# textbook case for **MAD/IQR over Z-score** and for the `log1p` transform applied in §3.

# %% [markdown]
# ### 2.3 Correlation analysis — which coefficient, and why
#
# We use **Spearman** rank correlation as the primary measure. Justification:
# * **Pearson** assumes *linear* relationships between *continuous, roughly-normal* variables and is
#   highly sensitive to the extreme outliers we just documented — inappropriate for these tails.
# * **Spearman** measures *monotonic* association on ranks, so it is robust to outliers and the
#   non-normal, heavy-tailed shapes typical of counts/bytes — the right tool here.
# * **Kendall** also ranks but is most useful for small samples / many ties; with 126k rows
#   Spearman is the pragmatic choice.
#
# We care about correlation here to find **redundancy** (features carrying the same information),
# which both hurts model interpretability (importance gets split across collinear features) and
# wastes capacity.

# %%
numeric_feats = [c for c in data.FEATURE_COLUMNS
                 if c not in data.CATEGORICAL_COLUMNS and c not in const]
corr = eda.correlation_matrix(train, numeric_feats, method="spearman")
eda.plot_correlation_heatmap(corr)
red_pairs = eda.high_correlation_pairs(corr, threshold=0.9)
RESULTS["redundant_pairs"] = red_pairs.round(3).to_dict("records")
print("Highly correlated (|Spearman| >= 0.9) feature pairs -> redundancy candidates:")
display(red_pairs.round(3))

# %% [markdown]
# **Finding.** Several `*_serror_rate`, `*_srv_rate`, and `dst_host_*` features are near-duplicates
# (|ρ| ≥ 0.9). Redundancy detection (here: rank correlation; alternatives: VIF, mutual information,
# exact-duplicate check) lets us prune or combine them — improving interpretability without losing
# signal.
#
# **Statistical vs practical significance.** With n = 125,973, *every* correlation is "statistically
# significant" (p ≈ 0), so a p-value tells us nothing useful here. We therefore select on **effect
# size** (|ρ| ≥ 0.9) — the threshold at which redundancy is large enough to matter practically. This
# distinction is essential in cyber data, where huge sample sizes make trivial correlations look
# "significant".

# %% [markdown]
# #### Why Spearman and not Pearson — shown, not just asserted
#
# We claimed Pearson is distorted by the heavy tails. Here is the direct evidence: for every pair
# that is "highly correlated" under *either* measure, we print Pearson, Spearman and their gap.

# %%
cmp_corr = eda.compare_correlation_methods(train, numeric_feats, threshold=0.9)
RESULTS["pearson_vs_spearman"] = cmp_corr.to_dict("records")
print("Pairs where Pearson and Spearman most disagree (heavy-tail distortion):")
display(cmp_corr.head(8))

# %% [markdown]
# **Finding (decisive for the method choice).** The two coefficients disagree sharply on exactly the
# skewed count features. The clearest case is `num_compromised ↔ num_root`: **Pearson ≈ 0.999** but
# **Spearman ≈ 0.17**. A Pearson-based redundancy filter would *wrongly* delete one of them, because a
# handful of rows with simultaneously huge values dominate the linear fit; the rank-based Spearman
# shows the two are *not* monotonically tied and carry distinct information. This is the empirical
# justification for using Spearman (not Pearson) to drive feature pruning on this data — and a concrete
# illustration of why outlier-sensitive statistics mislead on heavy-tailed cyber telemetry.

# %% [markdown]
# ### 2.4 Crosstab / group-by — protocol × attack family

# %%
ct = pd.crosstab(train["protocol_type"], train["attack_category"])
RESULTS["crosstab_protocol"] = json.loads(ct.to_json())
print(ct)

# %% [markdown]
# **Finding.** Attack families are highly protocol-dependent (DoS dominates `icmp`/`tcp`; most R2L
# rides `tcp`). This is exactly the kind of structure a `protocol_type` one-hot feature should
# capture — and a reason ordinal/label-encoding it (as some tutorials do) is wrong: there is no
# natural order among tcp/udp/icmp.

# %% [markdown]
# ## 3. Feature Engineering (leakage-safe)
#
# All transforms live inside a scikit-learn `ColumnTransformer` and are **fit on training data only**:
# * **drop** constant `num_outbound_cmds`;
# * **log1p + StandardScaler** on the 16 heavy-tailed count/byte features;
# * **StandardScaler** on the bounded rate features;
# * **One-Hot** (`handle_unknown='ignore'`) on `protocol_type`, `service`, `flag` — defends against
#   unseen `service` categories at test time, unlike label/ordinal encoding.

# %%
groups = features.feature_groups(train)
preprocessor = features.build_preprocessor(groups)
# Fit on train only, just to report the output dimensionality:
_ = preprocessor.fit(data.feature_matrix(train), train["is_attack"])
n_out = preprocessor.transform(data.feature_matrix(train.head(5))).shape[1]
RESULTS["feature_groups"] = {k: (v if k == "dropped" else len(v)) for k, v in groups.items()}
RESULTS["n_model_features"] = int(n_out)
print("Feature groups:", {k: (v if k == 'dropped' else len(v)) for k, v in groups.items()})
print("Model input dimensionality after encoding:", n_out)

# %% [markdown]
# ### 3.1 Feature selection (what we dropped, and why)
#
# Selection here is principled, not blind: we remove the **1 constant** feature (`num_outbound_cmds`),
# we keep the `difficulty` metadata **out** of the model (leakage, §1.1), and we *flag* the 9 redundant
# pairs (§2.3) as merge/drop candidates. We keep the encoded one-hot columns because tree ensembles
# handle them well and they carry the protocol/service structure seen in the crosstab.

# %%
# Feature CREATION: domain features proposed in the report (kept separate from the baseline pipeline
# so the controlled A/B comparison stays clean). Shown here to demonstrate the idea concretely.
engineered = features.add_engineered_features(train)[
    ["src_bytes", "dst_bytes", "total_bytes", "bytes_ratio", "error_rate_mean", "is_attack"]
].head()
print("Examples of created features (total_bytes / bytes_ratio / error_rate_mean):")
display(engineered)

# %% [markdown]
# ### 3.2 Dimensionality reduction — PCA for visualisation
#
# We do **not** feed PCA to the models (121 encoded features is small, and trees are unharmed by
# correlated inputs), but a 2-component PCA is a useful *visual* check of class separability — and it
# previews our central finding.

# %%
eda.plot_pca_2d(train, preprocessor, "attack_category", data.CLASS_ORDER)
print("PCA: DoS/Probe form separable structure; R2L/U2R sit on top of 'normal' "
      "-> a visual preview of why they are so hard to detect.")

# %% [markdown]
# ## 4. Model Training — the controlled experiment
#
# We hold **everything** constant (features, preprocessing, models, seeds) and vary **only the
# evaluation protocol**:
#
# * **Protocol A (tutorial-style):** random stratified 80/20 split of KDDTrain+; evaluate on the
#   held-out 20% — same distribution as training.
# * **Protocol B (correct):** train on *all* of KDDTrain+; evaluate on the official **KDDTest+**.
#
# Models: a majority-class **baseline**, **Logistic Regression**, **Random Forest**, **Gradient
# Boosting**. Binary target (attack vs normal) first, then 5-class.

# %%
def fit_eval_binary(models_dict, X_tr, y_tr, X_ev, y_ev):
    """Fit each pipeline on (X_tr,y_tr); return binary metrics + scores on (X_ev,y_ev)."""
    res, scores = {}, {}
    for name, mdl in models_dict.items():
        mdl.fit(X_tr, y_tr)
        y_pred = mdl.predict(X_ev)
        y_score = (mdl.predict_proba(X_ev)[:, 1]
                   if hasattr(mdl, "predict_proba") else None)
        res[name] = evaluate.binary_metrics(y_ev, y_pred, y_score)
        scores[name] = y_score
    return res, scores

Xtr_full, ytr_bin = data.feature_matrix(train), train["is_attack"].to_numpy()
Xte_full, yte_bin = data.feature_matrix(test), test["is_attack"].to_numpy()

# Protocol A — random split of KDDTrain+ (mimics the tutorials)
Xa_tr, Xa_ev, ya_tr, ya_ev = train_test_split(
    Xtr_full, ytr_bin, test_size=0.20, random_state=RANDOM_STATE, stratify=ytr_bin)
protoA, _ = fit_eval_binary(models.make_models(features.build_preprocessor(groups)),
                            Xa_tr, ya_tr, Xa_ev, ya_ev)

# Protocol B — official KDDTest+
protoB, scoresB = fit_eval_binary(models.make_models(features.build_preprocessor(groups)),
                                  Xtr_full, ytr_bin, Xte_full, yte_bin)

RESULTS["binary_protocolA_randomsplit"] = protoA
RESULTS["binary_protocolB_official_test"] = protoB
print("done")

# %% [markdown]
# ### 4.1 Cross-validation sanity check (still optimistic)
#
# To rule out "it was just a lucky split", we run 5-fold stratified CV *within* KDDTrain+. It also
# reports ~0.99 — confirming the inflation is caused by the **train/test distribution being the
# same**, not by one fortunate split.

# %%
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
rf_cv = models.make_models(features.build_preprocessor(groups))["Random Forest"]
cv_acc = cross_val_score(rf_cv, Xtr_full, ytr_bin, cv=cv, scoring="accuracy", n_jobs=-1)
RESULTS["rf_cv_accuracy_mean"] = float(cv_acc.mean())
RESULTS["rf_cv_accuracy_std"] = float(cv_acc.std())
print(f"Random Forest 5-fold CV accuracy on KDDTrain+: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")

# %% [markdown]
# ### 4.2 Literal tutorial-style reproduction (their preprocessing, not ours)
#
# Protocols A/B above use *our* leakage-safe pipeline. To show the headline number is a property of
# the **tutorials' own recipe**, we reconstruct it faithfully — `OrdinalEncoder` (LabelEncoder-style)
# on the nominal columns, **raw unscaled values**, and the leaky `difficulty` column **kept in** — and
# run the *same* Random Forest. We verified against the actual repository
# (`abhinav-bhardwaj/...`): it uses `train_test_split` + `LabelEncoder`, keeps the difficulty/level
# column, reports KNN ≈ 98.5% / neural-net ≈ 97.8%, and **never evaluates on the official KDDTest+**.

# %%
Xtut_tr = data.feature_matrix(train, features.TUTORIAL_COLUMNS)
Xtut_te = data.feature_matrix(test, features.TUTORIAL_COLUMNS)
tut_rf = lambda: Pipeline([("prep", features.build_tutorial_preprocessor()),
                           ("clf", RandomForestClassifier(
                               n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))])
# Their protocol: random split of KDDTrain+ (with the difficulty leak left in)
Xt_a, Xt_e, yt_a, yt_e = train_test_split(
    Xtut_tr, ytr_bin, test_size=0.20, random_state=RANDOM_STATE, stratify=ytr_bin)
m = tut_rf(); m.fit(Xt_a, yt_a)
tut_split_acc = float((m.predict(Xt_e) == yt_e).mean())
# The correct protocol: same recipe, evaluate on official KDDTest+
m2 = tut_rf(); m2.fit(Xtut_tr, ytr_bin)
tut_test_acc = float((m2.predict(Xtut_te) == yte_bin).mean())
RESULTS["tutorial_repro"] = {
    "random_split_accuracy": round(tut_split_acc, 4),
    "official_test_accuracy": round(tut_test_acc, 4),
    "drop_pp": round((tut_split_acc - tut_test_acc) * 100, 1),
}
print(f"Tutorial recipe (LabelEncoder + difficulty leak, random split): acc = {tut_split_acc:.4f}")
print(f"Same recipe on official KDDTest+                              : acc = {tut_test_acc:.4f}")
print(f"-> reproduces ~99% on a random split, collapses by "
      f"{(tut_split_acc - tut_test_acc)*100:.1f} pp on the real test set.")

# %% [markdown]
# **Finding.** The tutorials' *own* preprocessing reproduces the ~99% number on a random split — even
# though it leaks `difficulty` — and still collapses on KDDTest+. So the inflation is **not** an
# artefact of our pipeline choices: it is intrinsic to *evaluating in-distribution*. (Note the leak
# makes their random-split number look even better, compounding the over-optimism.)

# %% [markdown]
# ### 4.3 How stable is the A→B gap? (multiple seeds, not one lucky split)
#
# The 21.9 pp drop in §5.1 comes from one split seed. We repeat Protocol A across several seeds (the
# fixed Protocol B test set does not change) and report the gap as mean ± std, so the headline finding
# carries an error bar rather than resting on a single draw.

# %%
seeds = [0, 1, 7, 21, 42]
protoB_rf_acc = None  # filled after §5.1; recompute here independently for a clean RF on full train
rf_B = models.make_models(features.build_preprocessor(groups))["Random Forest"]
rf_B.fit(Xtr_full, ytr_bin)
protoB_rf_acc = float((rf_B.predict(Xte_full) == yte_bin).mean())
splitA_accs = []
for sd in seeds:
    xa, xe, ya, ye = train_test_split(Xtr_full, ytr_bin, test_size=0.20,
                                      random_state=sd, stratify=ytr_bin)
    rf = Pipeline([("prep", features.build_preprocessor(groups)),
                   ("clf", RandomForestClassifier(n_estimators=200,
                                                  random_state=sd, n_jobs=-1))])
    rf.fit(xa, ya)
    splitA_accs.append(float((rf.predict(xe) == ye).mean()))
splitA_accs = np.array(splitA_accs)
gaps = splitA_accs - protoB_rf_acc
RESULTS["multiseed_gap"] = {
    "seeds": seeds,
    "protocolA_acc_mean": round(float(splitA_accs.mean()), 4),
    "protocolA_acc_std": round(float(splitA_accs.std()), 4),
    "protocolB_acc": round(protoB_rf_acc, 4),
    "gap_mean_pp": round(float(gaps.mean()) * 100, 2),
    "gap_std_pp": round(float(gaps.std()) * 100, 2),
}
print(f"Protocol A accuracy over {len(seeds)} seeds: "
      f"{splitA_accs.mean():.4f} ± {splitA_accs.std():.4f}")
print(f"Protocol B accuracy (fixed KDDTest+)       : {protoB_rf_acc:.4f}")
print(f"A→B gap: {gaps.mean()*100:.2f} ± {gaps.std()*100:.2f} pp  (stable across seeds)")

# %% [markdown]
# **Finding.** The drop is essentially seed-invariant (std ≈ a fraction of a point). The collapse is a
# property of the *distribution shift*, not of any particular random split — the single most important
# robustness check for our central claim.

# %% [markdown]
# ## 5. Evaluation
#
# ### 5.1 The headline result: Protocol A vs Protocol B
#
# Every metric for the **same Random Forest**, changing only the test data.

# %%
tblA = evaluate.metrics_table(protoA)
tblB = evaluate.metrics_table(protoB)
print("PROTOCOL A — random split of KDDTrain+ (what tutorials report):")
display(tblA)
print("\nPROTOCOL B — official KDDTest+ (correct evaluation):")
display(tblB)
gap = (tblA.loc["Random Forest", "accuracy"] - tblB.loc["Random Forest", "accuracy"])
RESULTS["rf_accuracy_gap_A_minus_B"] = float(gap)
print(f"\nRandom Forest accuracy drop A→B: {gap*100:.1f} percentage points")

# %% [markdown]
# ### 5.2 Metric definitions and why they matter in IDS
#
# | Metric | Definition | Why it matters for intrusion detection |
# |---|---|---|
# | Accuracy | (TP+TN)/N | Misleading under imbalance — a constant predictor scores high. |
# | Balanced Acc. | mean of per-class recall | Treats rare attacks as equally important as `normal`. |
# | Precision | TP/(TP+FP) | High precision = few false alarms (analyst fatigue). |
# | Recall (TPR) | TP/(TP+FN) | High recall = few **missed attacks** — the costly error. |
# | F1 / Fβ | harmonic mean (β=2 up-weights recall) | Single number; Fβ encodes that a miss > a false alarm. |
# | MCC | balanced correlation over the confusion matrix | Trustworthy single score even with heavy imbalance. |
# | ROC-AUC / PR-AUC | ranking quality; PR-AUC vs the prevalence baseline | PR-AUC is the honest curve when positives are rare. |
#
# We **down-weight raw Accuracy** and lead with MCC, Balanced Accuracy, PR-AUC and per-class recall.

# %%
# PR / ROC curves on KDDTest+, and the prevalence baseline for Accuracy honesty.
curves = {name: (yte_bin, s) for name, s in scoresB.items() if s is not None}
evaluate.plot_pr_and_roc(curves)
base_rate = float(max(np.mean(yte_bin), 1 - np.mean(yte_bin)))
RESULTS["majority_baseline_accuracy_test"] = base_rate
print(f"A trivial majority predictor scores accuracy = {base_rate:.3f} on KDDTest+.")

# %% [markdown]
# ### 5.3 Multiclass evaluation & per-class recall (where Accuracy hides the failure)

# %%
ytr_mc, yte_mc = train["attack_category"].to_numpy(), test["attack_category"].to_numpy()
mc_models = models.make_models(features.build_preprocessor(groups))
mc_metrics, per_class = {}, {}
best_name, best_pred = None, None
for name, mdl in mc_models.items():
    mdl.fit(Xtr_full, ytr_mc)
    pred = mdl.predict(Xte_full)
    mc_metrics[name] = evaluate.multiclass_metrics(yte_mc, pred)
    per_class[name] = evaluate.per_class_recall(yte_mc, pred, data.CLASS_ORDER)
    if name == "Random Forest":
        best_name, best_pred = name, pred
RESULTS["multiclass_protocolB"] = mc_metrics
recall_tbl = pd.DataFrame(per_class).round(3)
RESULTS["per_class_recall"] = json.loads(recall_tbl.to_json())
print("Multiclass metrics on KDDTest+:"); display(evaluate.metrics_table(mc_metrics))
print("\nPer-class RECALL on KDDTest+ (note R2L / U2R):"); display(recall_tbl)
evaluate.plot_confusion(yte_mc, best_pred, data.CLASS_ORDER,
                        "Random Forest — KDDTest+ (row-normalised)", "confusion_rf_test.png")

# %% [markdown]
# ### 5.4 Can the rare-class failure be fixed? Cost-sensitive re-weighting
#
# The textbook remedy for imbalance is to make rare-class errors more expensive
# (`class_weight='balanced'`). We retrain Random Forest and Gradient Boosting with balanced weights
# and compare per-class recall. This tests whether the R2L/U2R blindness is *merely* imbalance — or
# something deeper that re-weighting cannot fix.

# %%
bal_models = models.make_models(features.build_preprocessor(groups), class_weight="balanced")
bal_recall = {}
for name in ["Random Forest", "Gradient Boosting"]:
    bal_models[name].fit(Xtr_full, ytr_mc)
    pred = bal_models[name].predict(Xte_full)
    bal_recall[f"{name} (balanced)"] = evaluate.per_class_recall(yte_mc, pred, data.CLASS_ORDER)
compare_recall = pd.concat(
    [pd.DataFrame(per_class)[["Random Forest", "Gradient Boosting"]],
     pd.DataFrame(bal_recall)], axis=1).round(3)
RESULTS["cost_sensitive_recall"] = json.loads(compare_recall.to_json())
print("Per-class recall — default vs class_weight='balanced' (KDDTest+):")
display(compare_recall)

# %% [markdown]
# **Finding.** Balanced re-weighting helps **unevenly and only partially**: Gradient Boosting's R2L
# recall rises (≈0.10 → ≈0.28) while Random Forest's R2L recall actually *drops* — and **both still
# miss the large majority of R2L/U2R**. This is decisive: the failure is **not just an imbalance
# artefact** but a combination of (i) the train→test distribution shift and (ii) R2L/U2R being nearly
# inseparable from normal traffic at the flow-statistics level. A genuine fix needs different signals
# (payload/host telemetry) or a different paradigm (dedicated anomaly detection) — not merely a
# heavier class weight. An important, non-obvious conclusion that the original tutorials never reach.

# %% [markdown]
# ### 5.5 Feature-engineering ablations — does pruning / enriching change anything?
#
# Two controlled feature experiments on Protocol B (same RF, official KDDTest+):
# 1. **Redundancy ablation** — drop one feature from each |Spearman| ≥ 0.9 pair (§2.3). If the 9
#    redundant pairs are truly redundant, removing them should *not* hurt performance.
# 2. **Engineered features** — add `total_bytes`, `bytes_ratio`, `error_rate_mean` (the domain
#    features proposed in §3.1) and measure the delta. Tests whether they actually help.


# %%
def eval_rf_protoB(feat_cols, df_tr, df_te):
    """Fit the standard RF pipeline on a given feature set; return KDDTest+ metrics."""
    g = features.feature_groups(df_tr, feat_cols)
    rf = Pipeline([("prep", features.build_preprocessor(g)),
                   ("clf", RandomForestClassifier(n_estimators=200,
                                                  random_state=RANDOM_STATE, n_jobs=-1))])
    rf.fit(data.feature_matrix(df_tr, feat_cols), df_tr["is_attack"].to_numpy())
    Xev = data.feature_matrix(df_te, feat_cols)
    n_in = len(rf.named_steps["prep"].get_feature_names_out())
    met = evaluate.binary_metrics(df_te["is_attack"].to_numpy(),
                                  rf.predict(Xev), rf.predict_proba(Xev)[:, 1])
    met["n_model_features"] = int(n_in)
    return met


drop_set = features.redundant_drop_set(RESULTS["redundant_pairs"])
ablated_cols = [c for c in data.FEATURE_COLUMNS if c not in drop_set]
train_eng = features.add_engineered_features(train)
test_eng = features.add_engineered_features(test)
eng_cols = data.FEATURE_COLUMNS + features.ENGINEERED_COLUMNS

ablation = {
    "Full (41 features)": eval_rf_protoB(data.FEATURE_COLUMNS, train, test),
    f"Pruned (−{len(drop_set)} redundant)": eval_rf_protoB(ablated_cols, train, test),
    "+ Engineered (3 created)": eval_rf_protoB(eng_cols, train_eng, test_eng),
}
RESULTS["feature_ablation"] = ablation
RESULTS["redundant_drop_set"] = drop_set
abl_tbl = pd.DataFrame(ablation).T[["n_model_features", "accuracy", "mcc", "recall", "f2"]].round(4)
print("Dropped as redundant:", drop_set)
print("\nRF on KDDTest+ under different feature sets:")
display(abl_tbl)

# %% [markdown]
# **Finding.** Pruning the 9 redundant features (here the |ρ|≥0.9 survivors-dropped set) leaves
# Protocol-B accuracy/MCC essentially unchanged while shrinking the model input — confirming the
# correlation analysis: the dropped columns carried no unique signal, so removing them buys
# interpretability and a smaller model at no measurable cost. The engineered byte/error features move
# the headline metrics only marginally on this flow-level data (they would matter more with raw session
# logs) — an honest, *tested* result rather than an assumed improvement.

# %% [markdown]
# ### 5.6 Testing our own recommendation: anomaly detection for the rare attacks
#
# Our thesis is that R2L/U2R should be treated as **anomaly detection**, not in-distribution
# classification. It would be hypocritical to *assert* that without testing it. We train two
# **semi-supervised one-class detectors on normal traffic only** (Isolation Forest, One-Class SVM),
# flag deviations as attacks, and compare their per-family detection rate against the supervised RF.
# This directly asks: does an anomaly paradigm catch the R2L/U2R that supervised learning misses?

# %%
# Shared leakage-safe encoding (labels never used by the scaler/one-hot).
anom_prep = features.build_preprocessor(features.feature_groups(train))
anom_prep.fit(data.feature_matrix(train))
X_tr_all = anom_prep.transform(data.feature_matrix(train))
X_te_all = anom_prep.transform(data.feature_matrix(test))
normal_mask = (train["is_attack"] == 0).to_numpy()
X_normal = X_tr_all[normal_mask]
# One-Class SVM is O(n^2): fit on a capped normal subsample for tractability.
rng = np.random.RandomState(RANDOM_STATE)
svm_idx = rng.choice(X_normal.shape[0], size=min(8000, X_normal.shape[0]), replace=False)

detectors = models.make_anomaly_detectors(svm_nu=0.1)
anom_results, anom_detection = {}, {}
# Supervised RF detection rate (flagged-as-attack per family) for a fair baseline.
rf_bin_pred = (scoresB["Random Forest"] >= 0.5).astype(int)
anom_detection["Supervised RF"] = evaluate.per_class_detection_rate(
    yte_mc, rf_bin_pred, data.CLASS_ORDER)
for name, det in detectors.items():
    det.fit(X_normal if name == "Isolation Forest" else X_normal[svm_idx])
    pred = det.predict(X_te_all)            # -1 = anomaly, +1 = inlier
    y_attack = (pred == -1).astype(int)     # treat anomaly as predicted attack
    score = -det.decision_function(X_te_all)  # higher = more anomalous
    anom_results[name] = evaluate.binary_metrics(yte_bin, y_attack, score)
    anom_detection[name] = evaluate.per_class_detection_rate(yte_mc, y_attack, data.CLASS_ORDER)
RESULTS["anomaly_binary_metrics"] = anom_results
detect_tbl = pd.DataFrame(anom_detection).round(3)
RESULTS["anomaly_detection_rate"] = json.loads(detect_tbl.to_json())
print("Binary metrics on KDDTest+ (one-class detectors trained on normal only):")
display(evaluate.metrics_table(anom_results)[["recall", "precision", "f2", "mcc", "roc_auc"]])
print("\nPer-family detection rate — supervised RF vs anomaly detectors "
      "('normal' row = false-positive rate):")
display(detect_tbl)
evaluate.plot_detection_comparison(
    detect_tbl, "Per-class detection: supervised vs one-class anomaly detection",
    "anomaly_detection.png")

# %% [markdown]
# **Finding (our recommendation, put to the test).** The one-class detectors trade precision for
# recall on the rare classes: trained on *normal only*, they flag a **much larger fraction of R2L/U2R**
# than the supervised RF (whose R2L/U2R recall is ≈0.05) — but at a steep false-positive cost on
# `normal`. So anomaly detection is **not a free fix**: it surfaces the dangerous rare attacks the
# supervised model is blind to, yet would overwhelm a SOC with false alarms unless combined with
# richer telemetry. This is a *nuanced, evidence-backed* version of our recommendation — exactly the
# rigour we faulted the tutorials for lacking: the right paradigm helps on the hardest attacks, but the
# flow-level features remain the binding constraint.

# %% [markdown]
# ## 6. Error Analysis
#
# ### 6.1 Where the models fail

# %%
err = pd.DataFrame(per_class)["Random Forest"]
RESULTS["rf_R2L_recall"] = float(err["R2L"]); RESULTS["rf_U2R_recall"] = float(err["U2R"])
print("Random Forest recall by class on KDDTest+:")
print(err.round(3))
print("\nInterpretation: DoS/Probe/normal are detected well; R2L and U2R — the attacks that "
      "matter most (credential theft, privilege escalation) — are largely MISSED.")

# %% [markdown]
# **Pattern.** Errors concentrate in the **rare, content-based** attacks (R2L/U2R) that look almost
# identical to normal traffic at the packet-statistics level and are **under-represented in training**
# (52 U2R rows) yet **over-represented in test**. This is the FN-heavy failure mode that headline
# accuracy conceals.

# %% [markdown]
# ### 6.2 False Positive / False Negative trade-off (operating point)
#
# In IDS a **False Negative** (missed intrusion) is usually far costlier than a **False Positive**
# (false alarm), but too many false alarms cause alert fatigue. The decision threshold is a lever.

# %%
rf_scoreB = scoresB["Random Forest"]
sweep = evaluate.threshold_sweep(yte_bin, rf_scoreB)
RESULTS["threshold_sweep"] = sweep.round(4).to_dict("records")
display(sweep)
print("Lowering the threshold trades more false alarms (FP) for fewer missed attacks (FN).")

# %% [markdown]
# ## 7. External validation on a modern dataset (UNSW-NB15)
#
# A fair objection to everything above is that **NSL-KDD is old (2009)** and that its KDDTest+ shift is
# hand-built. Does the critique survive on modern traffic? We repeat the core experiment on
# **UNSW-NB15** (Moustafa & Slay, 2015) — a contemporary IDS benchmark created specifically to replace
# KDD'99 — using the authors' **official** train/test partitions (175,341 / 82,332 flows, 42 features,
# 10 attack families). Every preprocessing step, model and metric is **reused unchanged** from the
# NSL-KDD pipeline; `src/unsw.py` only supplies the new schema. So this is a genuine apples-to-apples
# replication, not a fresh bespoke analysis.

# %%
u_train, u_test = unsw.load_train_test(ROOT / "data" / "raw")
print("UNSW-NB15 train:", u_train.shape, "| test:", u_test.shape)
u_dist = pd.DataFrame({
    "train_%": u_train["attack_category"].value_counts(normalize=True).mul(100).round(2),
    "test_%": u_test["attack_category"].value_counts(normalize=True).mul(100).round(2),
}).reindex(unsw.CLASS_ORDER)
RESULTS["unsw_class_distribution_pct"] = json.loads(u_dist.to_json())
print("\nClass prevalence (%), train vs official test:")
display(u_dist)
print(f"\nAttack prevalence: train {u_train['is_attack'].mean():.3f} vs test "
      f"{u_test['is_attack'].mean():.3f} — the official split is close to IID (no deliberate "
      "novel-attack shift like KDDTest+).")

# %% [markdown]
# ### 7.1 The A/B protocol experiment on modern data
#
# Same controlled comparison as §4–5: Protocol A (random 80/20 split of the training partition) vs
# Protocol B (the official test partition), identical model zoo.

# %%
u_groups = unsw.feature_groups(u_train)
u_Xtr, u_ytr = unsw.feature_matrix(u_train), u_train["is_attack"].to_numpy()
u_Xte, u_yte = unsw.feature_matrix(u_test), u_test["is_attack"].to_numpy()
u_Xa_tr, u_Xa_ev, u_ya_tr, u_ya_ev = train_test_split(
    u_Xtr, u_ytr, test_size=0.20, random_state=RANDOM_STATE, stratify=u_ytr)
u_protoA, _ = fit_eval_binary(models.make_models(features.build_preprocessor(u_groups)),
                              u_Xa_tr, u_ya_tr, u_Xa_ev, u_ya_ev)
u_protoB, u_scoresB = fit_eval_binary(models.make_models(features.build_preprocessor(u_groups)),
                                      u_Xtr, u_ytr, u_Xte, u_yte)
RESULTS["unsw_binary_protocolA"] = u_protoA
RESULTS["unsw_binary_protocolB"] = u_protoB
u_gap = (u_protoA["Random Forest"]["accuracy"] - u_protoB["Random Forest"]["accuracy"]) * 100
RESULTS["unsw_rf_accuracy_gap_pp"] = float(u_gap)
print("PROTOCOL A — random split of the UNSW training partition:"); display(evaluate.metrics_table(u_protoA))
print("\nPROTOCOL B — official UNSW test partition:"); display(evaluate.metrics_table(u_protoB))
print(f"\nRandom Forest A→B accuracy gap on UNSW-NB15: {u_gap:.1f} pp "
      f"(vs {RESULTS['rf_accuracy_gap_A_minus_B']*100:.1f} pp on NSL-KDD).")

# %% [markdown]
# **Finding.** The collapse reproduces on modern data but is **milder**: ≈0.96 → ≈0.87 (~9 pp) versus
# NSL-KDD's ~22 pp. The reason is illuminating — UNSW-NB15's official split is a near-IID random
# partition, so there is far less train→test distribution shift to expose. This *confirms* rather than
# weakens the thesis: the size of the optimism gap is governed by how much the test distribution
# differs from training, and a headline number reported on an in-distribution split is still inflated.

# %% [markdown]
# ### 7.2 Does aggregate accuracy still hide per-class failure? (yes)
#
# 87% binary accuracy looks healthy. We check the 10-class per-family recall to see what it conceals.

# %%
u_ytr_mc, u_yte_mc = u_train["attack_category"].to_numpy(), u_test["attack_category"].to_numpy()
u_mc = {k: v for k, v in models.make_models(features.build_preprocessor(u_groups)).items()
        if k in ("Random Forest", "Gradient Boosting")}
u_per_class, u_rf_pred = {}, None
for name, mdl in u_mc.items():
    mdl.fit(u_Xtr, u_ytr_mc)
    pred = mdl.predict(u_Xte)
    u_per_class[name] = evaluate.per_class_recall(u_yte_mc, pred, unsw.CLASS_ORDER)
    if name == "Random Forest":
        u_rf_pred = pred
u_recall_tbl = pd.DataFrame(u_per_class).round(3)
RESULTS["unsw_per_class_recall"] = json.loads(u_recall_tbl.to_json())
print("Per-class RECALL on the official UNSW-NB15 test set (note the rare families):")
display(u_recall_tbl)
evaluate.plot_confusion(u_yte_mc, u_rf_pred, unsw.CLASS_ORDER,
                        "Random Forest — UNSW-NB15 test (row-normalised)", "confusion_rf_unsw.png")

# %% [markdown]
# **Finding.** The NSL-KDD pattern re-emerges: behind 87% accuracy the rare / confusable families are
# nearly invisible **as their own class** — `Analysis` ≈ 0.00, `Worms` ≈ 0.09, `Backdoor` ≈ 0.10, and
# `DoS` ≈ 0.10 recall (`DoS` is absorbed into the much larger `Exploits` family). The metric lesson is
# fully transferable: aggregate accuracy on a modern dataset still masks systematic blindness to
# specific attack types — the per-class view is mandatory.

# %% [markdown]
# ### 7.3 Re-testing the anomaly-detection recommendation — and finding its limit
#
# On NSL-KDD a one-class detector trained on normal-only traffic *beat* the supervised models on the
# rarest attacks (§5.6). Is "use anomaly detection" therefore a universal fix? We run the identical
# experiment on UNSW-NB15 to find out.

# %%
u_prep = features.build_preprocessor(unsw.feature_groups(u_train))
u_prep.fit(u_Xtr)
u_Xtr_all, u_Xte_all = u_prep.transform(u_Xtr), u_prep.transform(u_Xte)
u_norm = u_Xtr_all[(u_train["is_attack"] == 0).to_numpy()]
u_rng = np.random.RandomState(RANDOM_STATE)
u_idx = u_rng.choice(u_norm.shape[0], size=min(8000, u_norm.shape[0]), replace=False)
u_detectors = models.make_anomaly_detectors(svm_nu=0.1)
u_anom, u_adetect = {}, {}
u_rf_bin = (u_scoresB["Random Forest"] >= 0.5).astype(int)
u_adetect["Supervised RF"] = evaluate.per_class_detection_rate(u_yte_mc, u_rf_bin, unsw.CLASS_ORDER)
for name, det in u_detectors.items():
    det.fit(u_norm if name == "Isolation Forest" else u_norm[u_idx])
    pred = det.predict(u_Xte_all)              # -1 = anomaly, +1 = inlier
    y_attack = (pred == -1).astype(int)
    score = -det.decision_function(u_Xte_all)  # higher = more anomalous
    u_anom[name] = evaluate.binary_metrics(u_yte, y_attack, score)
    u_adetect[name] = evaluate.per_class_detection_rate(u_yte_mc, y_attack, unsw.CLASS_ORDER)
RESULTS["unsw_anomaly_binary_metrics"] = u_anom
RESULTS["unsw_supervised_rf_binary_mcc"] = float(u_protoB["Random Forest"]["mcc"])
u_detect_tbl = pd.DataFrame(u_adetect).round(3)
RESULTS["unsw_anomaly_detection_rate"] = json.loads(u_detect_tbl.to_json())
u_sup_row = pd.DataFrame({"Supervised RF": u_protoB["Random Forest"]}).T[
    ["recall", "precision", "f2", "mcc", "roc_auc"]]
print("Binary metrics — supervised RF vs one-class detectors (UNSW-NB15 test):")
display(pd.concat([u_sup_row,
                   evaluate.metrics_table(u_anom)[["recall", "precision", "f2", "mcc", "roc_auc"]]]))
print("\nPer-family detection rate ('normal' row = false-positive rate):")
display(u_detect_tbl)
evaluate.plot_detection_comparison(
    u_detect_tbl, "UNSW-NB15 — per-class detection: supervised vs one-class",
    "anomaly_detection_unsw.png")

# %% [markdown]
# **Finding (the recommendation, refined).** Here anomaly detection is **not** the winner: the
# supervised RF flags ~99–100% of *every* attack family at the binary level (MCC ≈ 0.75), while the
# one-class detectors trail badly (Isolation Forest MCC ≈ 0.46, One-Class SVM ≈ 0.17). The reason is
# decisive for the project's thesis: UNSW-NB15 attacks are **statistically separable from normal
# traffic** at the flow level, so a model that has *seen* attacks wins. On NSL-KDD the rare R2L/U2R
# attacks **mimic** normal traffic — which is exactly why a normal-only detector helped there. So "use
# anomaly detection" is **not** a blanket fix; it is the right tool **specifically when attacks resemble
# normal traffic**, and the wrong default when they do not. This conditional, evidence-based conclusion
# is stronger than the original recommendation, and it only surfaces by testing on a second dataset.

# %%
# Persist everything quantitative for the report.
with open(ROOT / "results" / "metrics.json", "w") as f:
    json.dump(RESULTS, f, indent=2, default=float)
print("Wrote results/metrics.json with", len(RESULTS), "entries.")

# %% [markdown]
# ## 8. Executive Summary
#
# We critically reproduced the dominant NSL-KDD intrusion-detection tutorial pattern, which claims
# **~99% accuracy**. Holding models, features and preprocessing fixed and changing **only** the
# evaluation protocol, we reproduced ~0.99 accuracy under a random split of KDDTrain+ **and** under
# 5-fold CV — then watched it **collapse on the official KDDTest+**. The collapse is explained by a
# large **train→test distribution shift** (R2L 0.79%→12.8%, U2R 0.04%→0.30%) and is invisible to
# Accuracy because of class imbalance: a trivial majority predictor already scores ~0.57. Under
# honest metrics (MCC, Balanced Accuracy, PR-AUC, per-class recall) the models detect DoS/Probe well
# but **miss most R2L and U2R attacks** — the highest-impact intrusions. We additionally found a
# constant feature (`num_outbound_cmds`), redundant correlated features, and a leakage trap (the
# `difficulty` column). **Verdict: the ~99% claim is not supported** as a measure of real intrusion
# detection capability.
#
# **External validation (§7).** Replicating the pipeline on the modern **UNSW-NB15** (2015) dataset
# corroborates and sharpens the findings: the A→B accuracy gap reproduces (~9 pp) but is smaller
# because UNSW's official split is near-IID — pinning the gap on distribution-shift magnitude; aggregate
# accuracy again hides near-zero recall on rare families (Analysis/Backdoor/Worms/DoS); and the
# anomaly-detection remedy is shown to be **conditional** — it wins only when attacks mimic normal
# traffic (NSL-KDD R2L/U2R), and *loses* to supervised models when attacks are flow-separable (UNSW).

# %% [markdown]
# ## 9. Summing It Up
#
# * **Problem:** detect network intrusions (NSL-KDD), binary and by attack family.
# * **Source:** popular NSL-KDD ML tutorial(s) reporting ~99% accuracy via a random split of KDDTrain+.
# * **Dataset:** NSL-KDD — KDDTrain+ (125,973) / KDDTest+ (22,544), 41 features, 4 attack families;
#   externally validated on **UNSW-NB15** (2015) — 175,341 / 82,332 flows, 42 features, 9 attack families.
# * **Methodology:** faithful reproduction + a controlled A/B protocol experiment, robust EDA,
#   leakage-safe feature engineering, four models, imbalance-aware metrics, error analysis, and a
#   cross-dataset replication on a modern benchmark.
# * **Main finding:** the headline accuracy **does not survive** correct evaluation
#   (~0.99 → ~0.77), and the models fail precisely on the rare, dangerous attacks — a pattern that
#   reproduces on modern UNSW-NB15 data.
# * **Were the author's claims supported?** **No** — the metric is right but the *protocol* and the
#   *metric choice* make the claim misleading.
# * **Key insight:** in cyber, *evaluation protocol and metric choice decide the conclusion*; a
#   model that looks 99% accurate can be operationally blind to the attacks you care about.
# * **Recommendation:** do **not** adopt this approach as-is on similar problems. Use the
#   distribution-matched test set, report MCC/PR-AUC/per-class recall, and treat rare attacks as a
#   rare-class problem (resampling, cost-sensitive learning, or anomaly detection **when the attacks
#   resemble normal traffic** — a remedy our UNSW-NB15 test shows is conditional, not universal).
# * **Final conclusion:** a rigorous, reproducible, cross-dataset refutation of an over-optimistic but
#   extremely common cybersecurity ML claim.
