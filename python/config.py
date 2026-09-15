"""
Every analytic choice in Paper 6, in one place.

A parameter that is an assumption rather than an estimate lives here so it can
be changed in one line and so the robustness table can vary it.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MEPS = RAW / "meps"
DERIVED = ROOT / "data" / "derived"
TABLES = ROOT / "output" / "tables"
FIGURES = ROOT / "output" / "figures"
for _p in (DERIVED, TABLES, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 2026

# Single-PSU strata in variance estimation (svy.py): centre on the grand mean,
# as R's survey.lonely.psu = "adjust".
LONELY_PSU = "adjust"

# ---------------------------------------------------------------- data ----
# One two-year longitudinal file per panel, and the Medical Conditions file
# for that panel's first year. Panels 23 and 24 were later extended to three
# and four years; their original two-year files (HC-217, HC-225) are used so
# every person contributes exactly one prospective pair with a two-year weight.
PANELS = {
    23: {"file": "h217.dta", "year1": 2018, "conditions": "h207.dta"},
    24: {"file": "h225.dta", "year1": 2019, "conditions": "h214.dta"},
    25: {"file": "h234.dta", "year1": 2020, "conditions": "h222.dta"},
    26: {"file": "h244.dta", "year1": 2021, "conditions": "h231.dta"},
    27: {"file": "h252.dta", "year1": 2022, "conditions": "h241.dta"},
}

# CPI-U, all items, World Bank FP.CPI.TOTL, rebased to 2024 dollars; the same
# series used in Papers 3 and 4.
BASE_YEAR = 2024
CPI_TO_BASE = {2018: 1.24922, 2019: 1.22699, 2020: 1.21204, 2021: 1.15765,
               2022: 1.07187, 2023: 1.02950, 2024: 1.00000}

# Timing. MEPS rounds 1 and 2 fall in year 1; round 3 straddles the new year
# and is interviewed in year 2. Round-3 health items would therefore leak
# year-2 information into year-1 predictors. Perceived health is taken from
# round 2, ADL and IADL help from round 1 (the only year-1 round that asks).
HEALTH_ROUND = 2
FUNCTION_ROUND = 1

# People who died in year 2 have partial-year outcomes. Excluded from the
# primary sample; included with an indicator in robustness.
EXCLUDE_DIED = False
# Decedents' year-2 spending is annualized by their months in scope, so the
# outcome is a rate a payer would pay for the months of coverage. Excluding
# decedents is the robustness variant.
ANNUALIZE_DECEDENTS = True

# ------------------------------------------------------------- features ----
AGE_BANDS = [(0, 4), (5, 17), (18, 24), (25, 34), (35, 44), (45, 54),
             (55, 64), (65, 74), (75, 84), (85, 200)]
# A CCSR category becomes a flag if at least this share of the pooled sample
# (unweighted) carries it in year 1. Lever in robustness.
CCSR_MIN_PREVALENCE = 0.005

# Nested feature sets (spec Section 5.2).
#   F1 demographics; F2 + conditions; F3 + prior-year use and spending;
#   F4 + social-risk and function (income, insurance, health, ADL).
FEATURE_SETS = ("F1", "F2", "F3", "F4")
PRIMARY_FEATURE_SET = "F3"

# Outcome cap, as common in risk adjustment. Primary is uncapped; capped at
# this value in robustness.
OUTCOME_CAP = 250_000

# ------------------------------------------------------ cross-validation ----
# Folds are formed from primary sampling units within strata within panel,
# so no PSU contributes to both training and test data.
CV_FOLDS = 5
CV_REPEATS = 3

# ------------------------------------------------------------- fairness ----
# Evaluated groups. Race and ethnicity enter evaluation and fairness penalties
# only, never the feature matrix.
FAIRNESS_GROUPS = {
    "Income below 200% FPL": ("povcat", "le", 3),
    "Hispanic": ("race", "eq", 1),
    "Non-Hispanic Black": ("race", "eq", 3),
    "Non-Hispanic Asian": ("race", "eq", 4),
    "Needs ADL or IADL help": ("any_function_help", "eq", 1),
    "Mental health condition": ("any_mbd", "eq", 1),
    "Age 65 and over": ("age", "ge", 65),
    "Uninsured all year": ("inscov", "eq", 3),
    "Died in year 2": ("died", "eq", 1),
}
MIN_GROUP_N = 100

# Penalty weights for the fair-regression frontier (Zink and Rose 2020 form).
FAIR_LAMBDAS = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
# Groups whose net compensation the penalized and constrained models target.
FAIR_TARGET_GROUPS = ("Income below 200% FPL", "Mental health condition",
                      "Needs ADL or IADL help")

# ---------------------------------------------------- coding sensitivity ----
# Share of people who receive added condition flags, and how many.
CODING_SHARES = (0.05, 0.10, 0.25)
CODING_ADDED = (1, 2)
# Where added codes are drawn from: "prevalence" (proportional to observed
# prevalence among people with any condition) or "chronic" (restricted to the
# chronic categories listed in features.CHRONIC_CCSR).
CODING_POOLS = ("prevalence", "chronic", "targeted")

# ------------------------------------------------------------- plotting ----
FIG_DPI = 300
PALETTE = {"ink": "#1A1A1A", "muted": "#7F8C8D", "rule": "#D5D8DC",
           "linear": "#1B6CA8", "glm": "#117A65", "gbm": "#B03A2E",
           "nn": "#7D3C98", "fair": "#B7950B"}
