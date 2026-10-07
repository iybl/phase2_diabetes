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

