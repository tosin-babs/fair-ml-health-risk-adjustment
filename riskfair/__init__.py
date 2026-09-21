"""
riskfair: accuracy, fairness, interpretability and coding sensitivity for
health-cost prediction and risk adjustment, and, since 0.3.0, a strategic
plan to run any formula against and formulas robust to it.

    from riskfair import surveycv, metrics, models, fairness, coding, adversary, robust

    folds = surveycv.survey_folds(psu_ids, strata_ids)
    m = models.GBM("tweedie").fit(X[tr], y[tr], w[tr], clusters=psu_ids[tr])
    metrics.group_fairness(y[te], m.predict(X[te]), w[te], group_masks)

Nothing in the package reads a data file or a configuration: every choice is
an argument, so it runs on any person-level cost data with weights and design
identifiers, survey or claims.
"""

__version__ = "0.3.0"

from . import adversary, coding, fairness, metrics, models, robust, surveycv  # noqa: F401

__all__ = ["surveycv", "metrics", "models", "fairness", "coding", "adversary", "robust", "__version__"]
