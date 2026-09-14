# Interpretable and Fair Machine Learning for Health-Cost Prediction and Risk Adjustment

A reproducible benchmark of nine health-cost prediction models on public data,
evaluated together on accuracy, group fairness, interpretability and coding
sensitivity, and the `riskfair` Python package that runs the same evaluation on
any person-level cost data.

## Headline results

Medical Expenditure Panel Survey, two-year panels 23 to 27 (2018 to 2023),
43,568 persons, year-1 predictors and year-2 total spending in 2024 dollars.
Every number is out of sample from cross-validation with primary sampling units
kept whole.

| Quantity | Value |
|---|---:|
| R², weighted least squares (the payment-formula form) | 0.125 (0.101 to 0.158) |
| R², Tweedie LightGBM | 0.195 (0.161 to 0.245) |
| Source of the boosting advantage | prior use and spending; on diagnoses alone both reach about 0.10 |
| Underpayment of people needing ADL or IADL help, all models | $4,949 to $9,508 per person-year on a mean cost of $29,881 |
| Stacked constrained estimator: gap and R² | $210 and 0.178 (from $5,142 and 0.190) |
| Constrained linear formula: gap and R² | $805 and 0.096 (from $7,887 and 0.124) |
| Spillover of both constraints onto enrollees 65 and over | overpaid by $2,156 to $3,342 |
| Rise in payment per added chronic diagnosis code | WLS $1,292; Tweedie GLM $885; Tweedie LightGBM $693 |

## Interactive explorer

https://fair-risk-adjustment.vercel.app

Pick a group and see what each formula pays it relative to what it costs; move
the fairness penalty and watch accuracy and the gap; add diagnosis codes and see
which formula pays out most. It runs in the browser on the aggregate tables the
pipeline wrote; no person-level data leaves the pipeline.

## The package

```bash
pip install -e .          # from this directory; extras: .[neural] .[explain] .[dev]
```

```python
from riskfair import surveycv, models, metrics, fairness, coding

folds = surveycv.survey_folds(psu, stratum, n_splits=5, repeats=3)
m = models.GBM("tweedie").fit(X[tr], y[tr], w[tr], clusters=psu[tr])
metrics.accuracy(y[te], m.predict(X[te]), w[te])
metrics.group_fairness(y[te], m.predict(X[te]), w[te], {"low income": mask[te]})
fair = fairness.StackedFairGBM(constrained=True).fit(X[tr], y[tr], w[tr], [mask[tr]], clusters=psu[tr])
coding.stress(m.predict, X[te], w[te], columns, pool, weights, share=0.1, k=1, seed=0)
```

Nothing in the package reads a file or a configuration. It runs on survey or
claims data with weights and design identifiers; for claims with no survey
design, pass a household or provider identifier as the cluster.

## Data

| Source | Files | Access |
|---|---|---|
| MEPS two-year longitudinal files | HC-217, HC-225, HC-234, HC-244, HC-252 | meps.ahrq.gov, direct download |
| MEPS Medical Conditions files (year 1 of each panel) | HC-207, HC-214, HC-222, HC-231, HC-241 | meps.ahrq.gov |
| AHRQ CCSR reference file v2026.1 | DXCCSR-Reference-File-v2026-1.xlsx | hcup-us.ahrq.gov |

```bash
cd data/raw/meps
for h in h217 h225 h234 h244 h252 h207 h214 h222 h231 h241; do
  curl -sSL -o $h.zip https://meps.ahrq.gov/mepsweb/data_files/pufs/$h/${h}dta.zip && unzip -oq $h.zip
done
```

## Reproducing

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[neural,explain,dev]" pypdf
.venv/bin/python -m pytest -q tests            # package and leakage tests
.venv/bin/python python/run_all.py             # every table and figure, about 35 minutes
.venv/bin/python python/check_manuscript.py    # prose numbers against the tables
.venv/bin/python python/make_manuscript.py     # DOCX and PDF (needs pandoc)
```

`python/config.py` holds every parameter. Each pipeline step runs in its own
process: LightGBM, scikit-learn and PyTorch each ship an OpenMP runtime, and on
macOS loading them in one long-lived process deadlocks. On macOS, `brew install
libomp` is needed for LightGBM.

## Details that decide the results

- Predictors come only from year 1. Perceived health is taken from round 2 and
  functional help from round 1, because round 3 is interviewed in year 2.
- Race and ethnicity never enter a model; they are used only to evaluate what
  each model does to each group. A test asserts this.
- MEPS body-system placeholder codes (XXX000) are not CCSR categories and are
  excluded from condition flags and counts.
- Multiplicative models are rescaled to balance on their training data, the
  correction any payment system applies.
- The fair boosted model is stacked (a cross-fitted score inside a constrained
  regression). A net-compensation penalty inside the boosting objective was
  tried and rejected; the reason is in `riskfair/fairness.py`.
- SHAP values are for the squared-error LightGBM, which has the same accuracy as
  the Tweedie one and reports in dollars.

## Author

Oluwatosin Dorcas Babalola, Georgia State University, obabalola4@student.gsu.edu

## License

Code is MIT-licensed. MEPS public-use files are governed by AHRQ's data-use
terms and are not redistributed.
