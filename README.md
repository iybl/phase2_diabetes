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




