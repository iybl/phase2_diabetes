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
