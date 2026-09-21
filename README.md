# Interpretable and Fair Machine Learning for Health-Cost Prediction and Risk Adjustment

A reproducible benchmark of ten health-cost prediction models on public data,
evaluated together on accuracy, group fairness and its stability out of sample,
interpretability, and the response of payment to coding and to prior use, and
the `riskfair` Python package that runs the same evaluation on any person-level
cost data.

It is a research benchmark on survey data, not a payment formula.

## Headline results

Medical Expenditure Panel Survey, two-year panels 23 to 27 (2018 to 2023),
43,963 persons (people who died in year 2 kept, spending annualized and weighted
by exposure), year-1 predictors and year-2 total spending in 2024 dollars. Every
number is out of sample from cross-validation with primary sampling units kept
whole; intervals are Rao-Wu PSU bootstrap, and model comparisons are paired on
the same resamples.

| Quantity | Value |
|---|---:|
| R², unconstrained weighted least squares | 0.123 (0.102 to 0.155); 15.2% of predictions negative |
| R², payment-form model (sex-specific age cells, non-negative coefficients) | 0.107 (0.085 to 0.142) |
| R², Tweedie LightGBM | 0.187 (0.156 to 0.242); paired gain over WLS 0.064 (0.047 to 0.091) |
| Boosting gain without prior use and spending (F2) | −0.002 (−0.006 to 0.003) |
| Underpayment of people needing ADL or IADL help, all models | $4,545 to $9,583 per person-year on a mean cost of $31,329 |
| Constrained stacked estimator: largest target gap and R² | $296 (interval $365 to $4,590) and 0.171, from $4,601 and 0.182 |
| Constrained linear formula: largest target gap and R² | $724 ($220 to $5,515) and 0.099, from $7,747 and 0.123 |
| Out-of-fold ADL gap under the constraints, 15 folds | −$13,043 to +$6,534 |
| Spillover of the constraints | enrollees 65+ overpaid $2,242 to $3,362; uninsured underpaid up to $1,498 |
| Next-year payment per added dollar of year-1 spending (F3) | Tweedie LightGBM $0.34; WLS $0.28; payment-form $0.01 |
| Payment per added code, each model gamed on its own most lucrative codes (F2) | constrained WLS $17,469; payment-form $11,045; WLS $10,777; Tweedie LightGBM $5,510 |
| Same, random chronic codes (F2) | payment-form $4,327; Tweedie LightGBM $2,907; WLS $2,806 |

## Interactive explorer

https://fair-risk-adjustment.vercel.app

Pick a group and see what each formula pays it relative to what it costs; move
the fairness penalty and watch accuracy and the gap; add diagnosis codes and see
which formula pays out most. It runs in the browser on the aggregate tables the
pipeline wrote; no person-level data leaves the pipeline.

## The package

Version 0.3.0 adds two modules written for the follow-up paper
[adversarial-ml-risk-adjustment](https://github.com/tosin-babs/adversarial-ml-risk-adjustment):
`riskfair.adversary`, a plan that best-responds to any formula by coding and
by selection and a loop that trains a formula against it, and
`riskfair.robust`, payment-form formulas with a coding penalty, caps at
incremental cost, and a worst-subgroup (CVaR) penalty on expected mispricing.

```bash
pip install -e .          # from this directory; extras: .[neural] .[explain] .[dev]
```

```python
from riskfair import surveycv, models, metrics, fairness, coding, adversary, robust

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

## Authors

- Oluwatosin Dorcas Babalola, Department of Actuarial Science and Quantitative Risk Analysis and Management, Georgia State University, Atlanta, GA, USA, obabalola4@student.gsu.edu (corresponding)
- Chisom Adiegwu, Department of Actuarial Science and Quantitative Risk Analysis and Management, Georgia State University, Atlanta, GA, USA
- Eniola Zainab Olamilekan, Department of Actuarial Science and Quantitative Risk Analysis and Management, Georgia State University, Atlanta, GA, USA

## License

Code is MIT-licensed. MEPS public-use files are governed by AHRQ's data-use
terms and are not redistributed.
