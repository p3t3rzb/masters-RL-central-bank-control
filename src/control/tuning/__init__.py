"""Hyperparameter tuning for control policies, scored in the ground truth.

Every policy in this repository carries settings nobody has ever varied -- the
Taylor rule's two feedback gains are the plainest case, taken on faith from the
original paper at every call site. This package varies them, and does so against
the only judge that counts: the structural GROWTH model, run over the ensemble of
independent economies :mod:`data_generation` has written to disk.

The pieces:

* :mod:`~control.tuning.space` -- what may be varied, as value objects an
  optimiser is not needed to describe.
* :mod:`~control.tuning.policies` -- :class:`~control.tuning.policies.TunablePolicy`,
  the template that says how a draw of the hyperparameters becomes a runnable
  policy, split into hooks that fire once per candidate and once per economy so
  that a family whose candidates cost a training run fits the same interface as
  one whose candidates cost nothing.
* :mod:`~control.tuning.objective` -- what a candidate is worth: the mandate
  return it earns in the ground truth, averaged over a :class:`~control.tuning.objective.Fold`
  of economies, on a pool of processes.
* :mod:`~control.tuning.search` -- the study: an ordered walk of economies with
  pruning, and a separate validation fold that chooses the winner so the search's
  own optimism does not travel to the test split.

Run one with ``uv run python scripts/tune_policy.py --policy taylor``.
"""

from control.tuning.objective import (
    Evaluator,
    Fold,
    TuningConfig,
    make_fold,
    make_folds,
    score,
    score_group,
)
from control.tuning.policies import POLICIES, TaylorTuning, TunablePolicy
from control.tuning.search import TuningResult, optimise, score_params
from control.tuning.space import Categorical, Dimension, Integer, Real, SearchSpace

__all__ = [
    "Dimension",
    "Real",
    "Integer",
    "Categorical",
    "SearchSpace",
    "TunablePolicy",
    "TaylorTuning",
    "POLICIES",
    "TuningConfig",
    "Fold",
    "make_folds",
    "make_fold",
    "Evaluator",
    "score",
    "score_group",
    "optimise",
    "score_params",
    "TuningResult",
]
