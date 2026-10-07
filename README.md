# phase2_diabetes
# imports & setup
import os, sys, time
import importlib.metadata as md
import numpy as np
import pandas as pd
from scipy import stats
try:
    from skeLCS import eLCS
except ImportError as e:
    raise ImportError("scikit-eLCS is missing. Run  pip install scikit-eLCS  then restart the kernel.") from e
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt

DATA_PATH = os.environ.get("DATA_PATH", "diabetic_data.csv")
OUT_DIR = os.environ.get("OUT_DIR", "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

TEST_MODE = os.environ.get("ENGE707_TEST_MODE", "0") == "1"   # quick 1-fold test when set true
SEED = 42
N_SPLITS = 5
TOP_K_FEATURES = 12            # features kept by mutual information (improved system)
NEAR_CONSTANT_SHARE = 0.99     # drug columns that are "No" for >= 99% of training rows are dropped
BALANCE_SEED_BASE = 100        # balanced sample for fold k uses seed 100 + k
IMPROVED_ITERATIONS = 2000 if TEST_MODE else 60000
RUN_FULL_COHORT_SENSITIVITY = not TEST_MODE
MIN_GROUP_N = 200              # smallest subgroup reported in the fairness tables
MIN_GROUP_EVENTS = 20          # minimum positives AND negatives per subgroup
MIN_RULE_MATCHES = 50          # a rule needs this many held-out matches to be discussed
N_BOOT = 500                   # patient-level bootstrap resamples
EXPECTED_SHAPE = (101766, 50)
assert os.path.exists(DATA_PATH), f"{DATA_PATH} not found. Put diabetic_data.csv next to this file or set DATA_PATH."

versions = {"python": sys.version.split()[0]}
for pkg in ["numpy", "pandas", "scipy", "scikit-learn", "scikit-eLCS", "matplotlib"]:
    try:
        versions[pkg] = md.version(pkg)
    except md.PackageNotFoundError:
        versions[pkg] = "not found"
pd.Series(versions, name="version").to_csv(os.path.join(OUT_DIR, "environment_versions.csv"))
print("TEST_MODE:", TEST_MODE, "| improved iterations:", IMPROVED_ITERATIONS)
print(versions)

# load data  & apply the cohort rule 
# # "?" is the missing-value marker. Kept "None" as a category
# A1Cresult / max_glu_serum, means "test not performed", NOT missing (Phase I treated it as missing).
df_full = pd.read_csv(DATA_PATH, na_values=["?"], keep_default_na=False, low_memory=False)
print("raw data:", df_full.shape)
assert df_full.shape == EXPECTED_SHAPE, f"Unexpected dataset shape {df_full.shape}"
for _c in ["encounter_id", "patient_nbr", "readmitted", "discharge_disposition_id", "age", "diag_1", "diag_2", "diag_3"]:
    assert _c in df_full.columns, f"Column {_c} missing from the data file"
print("fully duplicated rows:", int(df_full.duplicated().sum()),
      "| duplicated encounter_id:", int(df_full["encounter_id"].duplicated().sum()),
      "| unique patients:", df_full["patient_nbr"].nunique())
print("A1Cresult:", df_full["A1Cresult"].value_counts(dropna=False).to_dict())
print("max_glu_serum:", df_full["max_glu_serum"].value_counts(dropna=False).to_dict())

# Patients who died or went to hospice cannot be readmitted, so they are excluded from the cohort.
cannot_return = ["11", "13", "14", "19", "20", "21"]
excl_mask = df_full["discharge_disposition_id"].astype(str).isin(cannot_return)
assert excl_mask.sum() > 0, "Death/hospice IDs not found"
print("\nremoved by discharge_disposition_id:",
      df_full.loc[excl_mask, "discharge_disposition_id"].astype(str).value_counts().to_dict(),
      "| total removed:", int(excl_mask.sum()))
pos_before = (df_full["readmitted"] == "<30").mean()
df = df_full[~excl_mask].reset_index(drop=True)
y = (df["readmitted"] == "<30").astype(int).to_numpy()
groups = df["patient_nbr"].to_numpy()
print(f"cohort: {df.shape} | positives: {int(y.sum())} ({y.mean():.4f}) | positive rate before exclusion: {pos_before:.4f}")
pd.DataFrame({
    "item": ["rows_raw", "rows_removed_death_hospice", "rows_cohort", "positives_raw", "positives_cohort",
             "pos_rate_raw", "pos_rate_cohort", "unique_patients_cohort"],
    "value": [len(df_full), int(excl_mask.sum()), len(df), int((df_full["readmitted"] == "<30").sum()),
              int(y.sum()), round(pos_before, 4), round(y.mean(), 4), int(pd.Series(groups).nunique())]
}).astype({"value": object}).to_csv(os.path.join(OUT_DIR, "cohort_summary.csv"), index=False)

# outlier check 
# Outliers are reported, not removed: high counts are plausible
num_cols = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
            "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
rows = []
for c in num_cols:
    s = pd.to_numeric(df[c], errors="coerce")
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo_b, hi_b = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out_mask = (s < lo_b) | (s > hi_b)
    rows.append({"feature": c, "median": s.median(), "p99": s.quantile(0.99), "max": s.max(),
                 "n_outside_1.5IQR": int(out_mask.sum()), "pct_outside": round(100 * out_mask.mean(), 2)})
outlier_table = pd.DataFrame(rows)
print(outlier_table.to_string(index=False))
outlier_table.to_csv(os.path.join(OUT_DIR, "outlier_check.csv"), index=False)



# Patient-grouped, stratified 5-fold split 
def make_folds(labels, grp):
    return list(StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
                .split(np.zeros((len(labels), 1)), labels, grp))

folds = make_folds(y, groups)
for tr, te in folds:
    assert not set(groups[tr]) & set(groups[te]), "Patient overlap between train and test!"
print("fold sizes (train, test):", [(len(tr), len(te)) for tr, te in folds])
print("positive rate per test fold:", [round(float(y[te].mean()), 4) for _, te in folds])
if TEST_MODE:
    folds = folds[:1]
    print("TEST_MODE: using fold 1 only")

# raw encoding helper 
# Minimal processing, drop identifiers and the target, and
# integer-label-encode text columns 
RAW_DROP = ["encounter_id", "patient_nbr", "readmitted"]
def label_encode(train_s, test_s):
    mapping = {v: i for i, v in enumerate(pd.unique(train_s.astype(object).dropna()))}
    n = len(mapping)
    a = train_s.astype(object).map(mapping).astype(float).fillna(n)
    b = test_s.astype(object).map(mapping).astype(float).fillna(n)
    return a, b, mapping

def encode_raw(frame, tr, te):
    train_out, test_out = frame.iloc[tr].copy(), frame.iloc[te].copy()
    for col in train_out.columns:
        if not pd.api.types.is_numeric_dtype(train_out[col]):
            train_out[col], test_out[col], _ = label_encode(train_out[col], test_out[col])
    return train_out.astype(float).to_numpy(), test_out.astype(float).to_numpy()

raw_df = df.drop(columns=RAW_DROP).copy()

# preprocessing and feature engineering
def icd_group(code):
    """Group an ICD-9 code into a broad clinical category (integer part: 250.83 -> 250)."""
    if pd.isna(code):
        return "Missing"
    code = str(code)
    if code.startswith(("V", "E")):
        return "Other"
    try:
        n = int(float(code))
    except ValueError:
        return "Other"
    if n == 250: return "Diabetes"
    if 390 <= n <= 459 or n == 785: return "Circulatory"
    if 460 <= n <= 519 or n == 786: return "Respiratory"
    if 520 <= n <= 579 or n == 787: return "Digestive"
    if 580 <= n <= 629 or n == 788: return "Genitourinary"
    if 710 <= n <= 739: return "Musculoskeletal"
    if 800 <= n <= 999: return "Injury"
    if 140 <= n <= 239: return "Neoplasms"
    return "Other"

discharge_map = {1: "Home", 6: "HomeHealth", 8: "HomeHealth",
                 3: "NursingOrRehab", 4: "NursingOrRehab", 22: "NursingOrRehab", 23: "NursingOrRehab",
                 24: "NursingOrRehab", 30: "NursingOrRehab", 15: "NursingOrRehab",
                 2: "OtherHospital", 5: "OtherHospital", 27: "OtherHospital", 28: "OtherHospital", 29: "OtherHospital",
                 7: "LeftAMA"}
source_map = {1: "Referral", 2: "Referral", 3: "Referral", 7: "Emergency",
              4: "Transfer", 5: "Transfer", 6: "Transfer", 10: "Transfer", 18: "Transfer",
              19: "Transfer", 22: "Transfer", 25: "Transfer", 26: "Transfer"}

df2 = df.copy()
df2["gender"] = df2["gender"].replace("Unknown/Invalid", "Unknown").fillna("Unknown")
df2["race"] = df2["race"].fillna("Unknown")
df2["age"] = df2["age"].str[1:].str.split("-").str[0].astype(int) + 5          # band midpoint, e.g. [70-80) -> 75
for c in ["diag_1", "diag_2", "diag_3"]:
    df2[c] = df2[c].map(icd_group)
df2["discharge_disposition_id"] = df2["discharge_disposition_id"].map(discharge_map).fillna("Other")
df2["admission_source_id"] = df2["admission_source_id"].map(source_map).fillna("Other")
df2["admission_type_id"] = df2["admission_type_id"].map(lambda v: "Unknown" if v in (5, 6, 8) else str(v))
for c in ["A1Cresult", "max_glu_serum"]:
    df2[c] = df2[c].replace("None", "NotMeasured").fillna("NotMeasured").astype(str)

df2 = df2.drop(columns=["encounter_id", "patient_nbr", "readmitted", "weight", "payer_code", "medical_specialty"])
print("after Phase II feature engineering:", df2.shape)
print("A1Cresult:", df2["A1Cresult"].value_counts().to_dict())
print("max_glu_serum:", df2["max_glu_serum"].value_counts().to_dict())
assert not df2.isna().any().any(), "NaN left after feature engineering"

drug_cols = [c for c in ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide",
                         "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
                         "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton", "insulin",
                         "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
                         "metformin-rosiglitazone", "metformin-pioglitazone"] if c in df2.columns]

def select_features_from_train(df_train):
    no_share = {c: (df_train[c] == "No").mean() for c in drug_cols}
    return [c for c, s in no_share.items() if s >= NEAR_CONSTANT_SHARE], no_share

def encode_elcs(train_df, test_df):
    train_out, test_out = train_df.copy(), test_df.copy()
    cat_cols = [c for c in train_out.columns if not pd.api.types.is_numeric_dtype(train_out[c])]
    mappings = {}
    for col in cat_cols:
        train_out[col], test_out[col], mappings[col] = label_encode(train_out[col], test_out[col])
    discrete_mask = np.array([c in cat_cols for c in train_out.columns], dtype=bool)
    return train_out.astype(float).to_numpy(), test_out.astype(float).to_numpy(), mappings, discrete_mask

def encode_conventional(train_df, test_df):
    cat_cols = [c for c in train_df.columns if not pd.api.types.is_numeric_dtype(train_df[c])]
    num_cols_ = [c for c in train_df.columns if c not in cat_cols]
    if not cat_cols:
        return train_df.astype(float).to_numpy(), test_df.astype(float).to_numpy(), None
    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    tr_cat = enc.fit_transform(train_df[cat_cols].astype(str))
    te_cat = enc.transform(test_df[cat_cols].astype(str))
    return (np.hstack([train_df[num_cols_].astype(float).to_numpy(), tr_cat]),
            np.hstack([test_df[num_cols_].astype(float).to_numpy(), te_cat]), enc)

def preprocess_fold(tr, te):
    base_train, base_test = df2.iloc[tr].copy(), df2.iloc[te].copy()
    near_constant, no_share = select_features_from_train(base_train)
    train_clean, test_clean = base_train.drop(columns=near_constant), base_test.drop(columns=near_constant)
    Xtr_e, Xte_e, mappings, discrete_mask = encode_elcs(train_clean, test_clean)
    Xtr_c, Xte_c, encoder = encode_conventional(train_clean, test_clean)
    for a in (Xtr_e, Xte_e, Xtr_c, Xte_c):
        assert np.isfinite(a).all(), "NaN/inf in encoded data"
    return {"Xtr_e": Xtr_e, "Xte_e": Xte_e, "Xtr_c": Xtr_c, "Xte_c": Xte_c, "mappings": mappings,
            "discrete_mask": discrete_mask, "near_constant": near_constant, "no_share": no_share,
            "train_clean": train_clean, "test_clean": test_clean, "encoder": encoder}

def top_k_by_mutual_info(Xtr, ytr, discrete_mask, k=TOP_K_FEATURES):
    mi = mutual_info_classif(Xtr, ytr, discrete_features=discrete_mask, random_state=SEED)
    return list(np.argsort(mi)[::-1][:k])

# class-balancing helper 
def balanced_sample(idx, labels, rng):
    idx = np.asarray(idx); labels = np.asarray(labels)
    pos, neg = idx[labels[idx] == 1], idx[labels[idx] == 0]
    assert len(neg) >= len(pos), "Fewer negatives than positives, cannot balance 1:1"
    return rng.permutation(np.concatenate([pos, rng.choice(neg, size=len(pos), replace=False)]))

# sanity check on fold 1
tr, te = folds[0]
p = preprocess_fold(tr, te)
print("eLCS train/test:", p["Xtr_e"].shape, p["Xte_e"].shape)
print("one-hot train/test:", p["Xtr_c"].shape, p["Xte_c"].shape)
print("drug columns dropped:", len(p["near_constant"]))
bal = balanced_sample(np.arange(len(tr)), y[tr], np.random.default_rng(BALANCE_SEED_BASE))
assert len(set(groups[tr]) & set(groups[te])) == 0
assert p["Xtr_e"].shape[1] == p["Xte_e"].shape[1] and p["Xtr_c"].shape[1] == p["Xte_c"].shape[1]
assert abs(y[tr][bal].mean() - 0.5) < 1e-9, "Balanced sample is not 50/50"
print("balanced rows:", len(bal), "| positive share:", y[tr][bal].mean())

 # main experiment: all models, all folds (slow) 
def metrics(yt_, pred_, proba_):
    return {"acc": accuracy_score(yt_, pred_), "bal_acc": balanced_accuracy_score(yt_, pred_),
            "precision": precision_score(yt_, pred_, zero_division=0), "recall": recall_score(yt_, pred_),
            "F1": f1_score(yt_, pred_), "ROC-AUC": roc_auc_score(yt_, proba_),
            "PR-AUC": average_precision_score(yt_, proba_)}

MAIN = ["eLCS raw", "eLCS cleaned", "eLCS balanced (10k)", "eLCS balanced + top12 (10k)", "eLCS improved",
        "Logistic regression", "Decision tree", "Random forest"]
MATCHED = ["Logistic regression (matched)", "Decision tree (matched)", "Random forest (matched)"]

def conventional_models():
    return {"Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=SEED)),
            "Decision tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=100, class_weight="balanced", random_state=SEED),
            "Random forest": RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=50,
                                                    class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)}

def matched_models():   # same balanced rows and same 12 features as the improved eLCS, no class weights
    return {"Logistic regression (matched)": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000, random_state=SEED)),
            "Decision tree (matched)": DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=SEED),
            "Random forest (matched)": RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=50, n_jobs=-1, random_state=SEED)}

all_rows, oof, fold_artifacts, fold1_models = [], {}, {}, {}

def fit_and_score(k, te, name, model, Xa, ya, Xb):
    t0 = time.time(); model.fit(Xa, ya); fit_s = time.time() - t0
    t1 = time.time(); pred_ = model.predict(Xb); proba_ = model.predict_proba(Xb)[:, 1]; pred_s = time.time() - t1
    all_rows.append({"fold": k + 1, "model": name, "fit_seconds": round(fit_s, 1),
                     "predict_seconds": round(pred_s, 1), **metrics(y[te], pred_, proba_)})
    oof.setdefault(name, []).append((te, pred_, proba_))
    if k == 0 and name == "eLCS improved":
        fold1_models[name] = model            # kept so Cell 12 explains the SAME model that was scored
    return model

def fit_elcs(k, te, name, Xa, ya, Xb, **kw):
    return fit_and_score(k, te, name, eLCS(random_state=k + 1, **kw), Xa, ya, Xb)

t_all = time.time()
for k, (tr, te) in enumerate(folds):
    ytr = y[tr]
    prep = preprocess_fold(tr, te)
    Xtr_e, Xte_e, Xtr_c, Xte_c = prep["Xtr_e"], prep["Xte_e"], prep["Xtr_c"], prep["Xte_c"]

    raw_tr, raw_te = encode_raw(raw_df, tr, te)
    fit_elcs(k, te, "eLCS raw", raw_tr, ytr, raw_te)
    fit_elcs(k, te, "eLCS cleaned", Xtr_e, ytr, Xte_e)
    bal_local = balanced_sample(np.arange(len(tr)), ytr, np.random.default_rng(BALANCE_SEED_BASE + k))
    fit_elcs(k, te, "eLCS balanced (10k)", Xtr_e[bal_local], ytr[bal_local], Xte_e)
    top_k = top_k_by_mutual_info(Xtr_e, ytr, prep["discrete_mask"])
    prep["top_k"], prep["bal_idx"] = top_k, bal_local
    fold_artifacts[k] = prep
    fit_elcs(k, te, "eLCS balanced + top12 (10k)", Xtr_e[bal_local][:, top_k], ytr[bal_local], Xte_e[:, top_k])
    fit_elcs(k, te, "eLCS improved", Xtr_e[bal_local][:, top_k], ytr[bal_local], Xte_e[:, top_k],
             learning_iterations=IMPROVED_ITERATIONS)

    for name, mod in conventional_models().items():
        fit_and_score(k, te, name, mod, Xtr_c, ytr, Xte_c)

    feat_cols = list(prep["train_clean"].columns[top_k])
    Xtr_m, Xte_m, _ = encode_conventional(prep["train_clean"].iloc[bal_local][feat_cols], prep["test_clean"][feat_cols])
    for name, mod in matched_models().items():
        fit_and_score(k, te, name, mod, Xtr_m, ytr[bal_local], Xte_m)

    pd.DataFrame(all_rows).to_csv(os.path.join(OUT_DIR, "fold_results.csv"), index=False)   # checkpoint after every fold
    print(f"fold {k + 1} done ({round(time.time() - t_all)} s elapsed, checkpoint saved) | dropped near-constant drugs: {prep['near_constant']}")

fold_results = pd.DataFrame(all_rows)
fold_results.to_csv(os.path.join(OUT_DIR, "fold_results.csv"), index=False)
print(fold_results[["model", "acc", "bal_acc", "recall", "ROC-AUC", "PR-AUC", "fit_seconds", "predict_seconds"]].round(3).to_string(index=False))

# results table and full-cohort sensitivity run
summary = fold_results.drop(columns="fold").groupby("model").agg(["mean", "std"]).round(3)
summary.to_csv(os.path.join(OUT_DIR, "summary_means.csv"))
ALL_MODELS = [m for m in MAIN + MATCHED if m in oof]

display_names = {"eLCS raw": "Original eLCS (raw)", "eLCS cleaned": "Original eLCS (preprocessed)",
                 "eLCS balanced (10k)": "Balanced eLCS", "eLCS balanced + top12 (10k)": "Feature-selected eLCS",
                 "eLCS improved": "Improved eLCS", "Logistic regression": "Logistic Regression",
                 "Decision tree": "Decision Tree", "Random forest": "Random Forest"}
cols = ["bal_acc", "ROC-AUC", "PR-AUC", "precision", "recall", "F1", "acc"]
means = fold_results.groupby("model")[cols].mean()
sds = fold_results.groupby("model")[cols].std()
results_table = pd.DataFrame({c: [f"{means.loc[m, c]:.3f} ({sds.loc[m, c]:.3f})" for m in ALL_MODELS] for c in cols},
                             index=[display_names.get(m, m) for m in ALL_MODELS])
results_table.columns = ["Balanced Accuracy", "ROC-AUC", "PR-AUC", "Precision", "Recall", "F1", "Accuracy"]
results_table.to_csv(os.path.join(OUT_DIR, "results_table_mean_sd.csv"))
print("fold mean (sd):")
print(results_table.to_string())

if RUN_FULL_COHORT_SENSITIVITY:
    y_all = (df_full["readmitted"] == "<30").astype(int).to_numpy()
    g_all = df_full["patient_nbr"].to_numpy()
    raw_all = df_full.drop(columns=RAW_DROP)
    sens = []
    for k, (tr_, te_) in enumerate(make_folds(y_all, g_all)):
        assert not set(g_all[tr_]) & set(g_all[te_])
        a, b = encode_raw(raw_all, tr_, te_)
        m = eLCS(random_state=k + 1); m.fit(a, y_all[tr_])
        sens.append({"fold": k + 1, "model": "eLCS raw (full cohort, no exclusion)",
                     **metrics(y_all[te_], m.predict(b), m.predict_proba(b)[:, 1])})
    sens = pd.DataFrame(sens)
    sens.to_csv(os.path.join(OUT_DIR, "sensitivity_raw_full_cohort.csv"), index=False)
    print("\nSensitivity, raw eLCS on the full cohort (mean over folds):")
    print(sens.drop(columns=["fold", "model"]).mean().round(3).to_dict())

# statistical tests: Friedman, corrected t-test, Holm
TEST_METRICS = ["PR-AUC", "ROC-AUC", "bal_acc", "recall", "F1"]
k_folds = len(folds)
ratio = np.mean([len(te) / len(tr) for tr, te in folds])

def nb_test(a, b):
    """Nadeau-Bengio corrected resampled paired t-test on per-fold metric differences."""
    d = (a - b).to_numpy(); var = d.var(ddof=1)
    if var == 0:
        return d.mean(), 1.0
    t = d.mean() / np.sqrt((1 / k_folds + ratio) * var)
    return d.mean(), 2 * stats.t.sf(abs(t), df=k_folds - 1)

def holm(p):
    p = np.asarray(p, dtype=float); order = np.argsort(p); adj = np.empty(len(p)); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i]); adj[i] = min(1.0, running)
    return adj

def pivot(metric):
    return fold_results.pivot(index="fold", columns="model", values=metric)

if k_folds < 3:
    print("TEST_MODE (fewer than 3 folds): statistical tests skipped. Run the full 5-fold experiment for Cell 10.")
else:
    fr = []
    for metric in TEST_METRICS:
        piv = pivot(metric)[MAIN]
        chi2, p_f = stats.friedmanchisquare(*[piv[m] for m in MAIN])
        fr.append({"metric": metric, "chi2": round(chi2, 3), "p": round(p_f, 5)})
        print(f"Friedman ({metric}): chi2 = {chi2:.2f}, p = {p_f:.5f}")
    pd.DataFrame(fr).to_csv(os.path.join(OUT_DIR, "stats_friedman.csv"), index=False)
    others = [m for m in ALL_MODELS if m != "eLCS improved"]
    tabs = []
    for metric in TEST_METRICS:
        piv = pivot(metric); res = [nb_test(piv["eLCS improved"], piv[o]) for o in others]
        tabs.append(pd.DataFrame({f"{metric} diff": [r[0] for r in res],
                                  f"{metric} p (Holm)": holm([r[1] for r in res])}, index=others))
    vs_others = pd.concat(tabs, axis=1).round(4)
    vs_others.to_csv(os.path.join(OUT_DIR, "stats_improved_vs_others.csv"))
    print("\nImproved eLCS vs each other model (diff > 0 means improved eLCS is higher):")
    print(vs_others.to_string())
    chain = ["eLCS raw", "eLCS cleaned", "eLCS balanced (10k)", "eLCS balanced + top12 (10k)", "eLCS improved"]
    tabs = []
    for metric in TEST_METRICS:
        piv = pivot(metric); res = [nb_test(piv[b_], piv[a_]) for a_, b_ in zip(chain[:-1], chain[1:])]
        tabs.append(pd.DataFrame({f"{metric} diff": [r[0] for r in res],
                                  f"{metric} p (Holm)": holm([r[1] for r in res])}, index=[f"+ {b_}" for b_ in chain[1:]]))
    ablation = pd.concat(tabs, axis=1).round(4)
    ablation.to_csv(os.path.join(OUT_DIR, "stats_ablation.csv"))
    print("\nAblation (each step vs the previous one):")
    print(ablation.to_string())

# confusion matrices (pooled over the test folds)
def pooled(name):
    return (np.concatenate([t for t, _, _ in oof[name]]), np.concatenate([p_ for _, p_, _ in oof[name]]),
            np.concatenate([b for _, _, b in oof[name]]))

cm_rows = []
for name in ALL_MODELS:
    te_, pr_, _ = pooled(name)
    cm = confusion_matrix(y[te_], pr_, labels=[0, 1])
    print(f"\n{name} (rows = actual 0/1, columns = predicted 0/1)\n{cm}")
    cm_rows.append({"model": name, "TN": cm[0, 0], "FP": cm[0, 1], "FN": cm[1, 0], "TP": cm[1, 1]})
pd.DataFrame(cm_rows).to_csv(os.path.join(OUT_DIR, "confusion_matrices.csv"), index=False)

# LCS rule extraction (fold 1, the SAME improved model) 
tr, te = folds[0]
art = fold_artifacts[0]
rule_model = fold1_models["eLCS improved"]
top_k = art["top_k"]
Xtr_sel, Xte_sel, yte = art["Xtr_e"][:, top_k], art["Xte_e"][:, top_k], y[te]
feat_names = [art["train_clean"].columns[j] for j in top_k]
labels = {c: list(m.keys()) for c, m in art["mappings"].items()}
uniq_vals = [np.unique(Xtr_sel[:, a]) for a in range(Xtr_sel.shape[1])]
print("Selected features:", feat_names)
pd.Series(feat_names, name="top12_features_fold1").to_csv(os.path.join(OUT_DIR, "selected_features_fold1.csv"), index=False)

def fmt(v):
    return str(int(v)) if float(v).is_integer() else f"{v:.2f}"

def inside(col, cond):                       # strict inequalities, same test as skeLCS matching
    return (col > cond[0]) & (col < cond[1])

def describe_condition(a, cond):
    """Readable text for one rule condition, or None when it excludes nothing (no information)."""
    name = feat_names[a]
    if isinstance(cond, (list, tuple, np.ndarray)):
        if name in labels:                   # category with >10 levels: interval over arbitrary integer codes
            members = [str(labels[name][c]) for c in range(len(labels[name])) if cond[0] < c < cond[1]]
            return None if len(members) == len(labels[name]) else f"{name} in [{', '.join(members)}]"
        if inside(Xtr_sel[:, a], cond).mean() >= 0.99:
            return None
        vals = uniq_vals[a]
        vin = vals[(vals > cond[0]) & (vals < cond[1])]       # observed values the interval really accepts
        if len(vin) == 0:
            return f"{name} matches no observed value"
        lo_open, hi_open = vin.min() == vals.min(), vin.max() == vals.max()
        if lo_open and hi_open: return None
        if lo_open: return f"{name} <= {fmt(vin.max())}"
        if hi_open: return f"{name} >= {fmt(vin.min())}"
        return f"{name} = {fmt(vin.min())}" if vin.min() == vin.max() else f"{fmt(vin.min())} <= {name} <= {fmt(vin.max())}"
    return f"{name} = {labels[name][int(cond)] if name in labels else fmt(cond)}"

def describe_rule(rule):
    parts = [t for t in (describe_condition(a, c) for a, c in zip(rule.specifiedAttList, rule.condition)) if t]
    return " AND ".join(parts) if parts else "(always true)"

def rule_matches(rule, Xm):
    ok = np.ones(len(Xm), dtype=bool)
    for a, cond in zip(rule.specifiedAttList, rule.condition):
        col = Xm[:, a]
        ok &= inside(col, cond) if isinstance(cond, (list, tuple, np.ndarray)) else (col == cond)
    return ok

base = yte.mean()
rows, covered_any, covered_informative = [], np.zeros(len(Xte_sel), bool), np.zeros(len(Xte_sel), bool)
for rule in rule_model.population.popSet:
    m_ = rule_matches(rule, Xte_sel); n_m = int(m_.sum()); text = describe_rule(rule)
    covered_any |= m_
    if text != "(always true)":
        covered_informative |= m_
    rows.append({"rule": text, "predicts": int(rule.phenotype), "numerosity": int(rule.numerosity),
                 "train_acc": rule.accuracy, "test_matches": n_m,
                 "test_readmit_rate": yte[m_].mean() if n_m else np.nan})
raw_rules = pd.DataFrame(rows)
rules = (raw_rules.groupby(["rule", "predicts"], as_index=False)
         .agg(numerosity=("numerosity", "sum"), n_merged=("numerosity", "size"), train_acc=("train_acc", "mean"),
              test_matches=("test_matches", "mean"), test_readmit_rate=("test_readmit_rate", "mean")))
n_always = int(rules.loc[rules.rule == "(always true)", "n_merged"].sum())
rules = rules[rules.rule != "(always true)"].reset_index(drop=True)      # no conditions = no information
rules["lift"] = rules["test_readmit_rate"] / base
rules = rules.round(3)
rules.to_csv(os.path.join(OUT_DIR, "rules_fold1.csv"), index=False)

unmatched = ~covered_any
fallback = np.bincount(rule_model.predict(Xte_sel[unmatched]).astype(int), minlength=2) if unmatched.any() else np.array([0, 0])
cov = pd.DataFrame({"item": ["test_rows", "matched_by_any_rule", "matched_by_informative_rule", "unmatched_rows",
                             "unmatched_predicted_0", "unmatched_predicted_1", "always_true_rules_removed", "distinct_informative_rules"],
                    "value": [len(Xte_sel), int(covered_any.sum()), int(covered_informative.sum()), int(unmatched.sum()),
                              int(fallback[0]), int(fallback[1]), n_always, len(rules)]})
cov.to_csv(os.path.join(OUT_DIR, "rule_coverage_fold1.csv"), index=False)
print(cov.to_string(index=False))
print("Base readmission rate in the test fold:", round(base, 3))
print("NOTE: train_acc is measured on the BALANCED training sample; compare rules using test match rate and lift.")

pd.set_option("display.max_colwidth", None); pd.set_option("display.width", 250)
for cls, title in [(1, "Rules predicting READMITTED"), (0, "Rules predicting NOT readmitted")]:
    print("\n" + title + f" (top by numerosity, at least {MIN_RULE_MATCHES} held-out matches)")
    print(rules[(rules.predicts == cls) & (rules.test_matches >= MIN_RULE_MATCHES)].sort_values("numerosity", ascending=False).head(8).to_string(index=False))
print("\nRules that test number_inpatient (best lift first):")
mask = rules["rule"].str.contains("number_inpatient") & (rules.test_matches >= MIN_RULE_MATCHES)
print(rules[mask].sort_values("lift", ascending=False).head(6).to_string(index=False))






