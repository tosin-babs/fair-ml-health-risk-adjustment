"""
Bridge between the paper's pipeline and the riskfair package.

The package holds everything reusable and reads no configuration. This module
binds it to this paper's configuration, so pipeline scripts get folds, models
and the fixed seed from one place.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from riskfair import coding, fairness, metrics, models, surveycv  # noqa: E402,F401


def folds(clusters, strata, repeats=None):
    return surveycv.survey_folds(clusters, strata, n_splits=config.CV_FOLDS,
                                 repeats=repeats or config.CV_REPEATS,
                                 seed=config.SEED)


def registry():
    return models.registry(seed=config.SEED)
