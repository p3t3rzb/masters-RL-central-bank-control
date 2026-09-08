"""The study: proposing candidates, abandoning bad ones, and choosing the winner.

The search walks an ordered sequence of economies rather than a fixed sample of
them. A candidate is scored on the first block of the sequence, reports what it
has earned so far, and is either allowed to continue onto the next block or
abandoned. Two things fall out of that arrangement, and both matter more than
which sampler is in use:

* every candidate is compared against every other on the **same prefix** of the
  same economies in the same order, so a difference between two losses is a
  difference between two policies and not between two draws;
* a hopeless candidate costs one block while a promising one is scored on tens of
  economies, which is how the sequence stays long without the study's cost growing
  with its length.

What the search *cannot* do is pick the winner. The best of many noisy estimates
is optimistic by construction -- the more candidates tried, the more the best one
owes to luck -- so :func:`optimise` re-scores its finalists on a validation fold
of economies the sequence never reached, and returns the winner of *that*. The
search's own best value is reported alongside, as a diagnostic rather than as an
answer: the gap between the two is the size of the bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler

from control.tuning.objective import Evaluator, Fold, TuningConfig
from control.tuning.policies import TunablePolicy


# -- the result ------------------------------------------------------------


@dataclass
class TuningResult:
    """One finished study: every candidate tried, and the one that was chosen.

    ``trials_`` holds each candidate with the loss and depth it reached on the
    search sequence; ``finalists_`` the few that were re-scored on the validation
    fold, each carrying both numbers; ``best_`` the finalist that won *there*.
    """

    policy: str
    space: dict[str, dict[str, Any]]
    config: TuningConfig
    trials_: list[dict[str, Any]] = field(default_factory=list)
    finalists_: list[dict[str, Any]] = field(default_factory=list)
    best_: dict[str, Any] = field(default_factory=dict)

    @property
    def best_params(self) -> dict[str, Any]:
        """The winning hyperparameters."""
        return dict(self.best_["params"])


# -- scoring a named configuration -----------------------------------------


def score_params(
    policy: TunablePolicy,
    params: Mapping[str, Any],
    fold: Fold,
    config: TuningConfig,
    *,
    evaluator: Evaluator | None = None,
) -> dict[str, Any]:
    """Score one fully-specified configuration over a whole fold.

    The un-pruned form of what a trial does, for the two places a complete score is
    wanted rather than a search: re-scoring a finalist, and scoring a named
    reference such as the untuned defaults. Reuses ``evaluator``'s warm pool when
    one is passed.
    """
    owned = evaluator is None
    evaluator = evaluator or Evaluator(config)
    try:
        prepared = policy.prepare(params, fold, config)
        return evaluator.score(policy, prepared, fold)
    finally:
        if owned:
            evaluator.close()


# -- the study -------------------------------------------------------------


def optimise(
    policy: TunablePolicy,
    fold: Fold,
    val_fold: Fold,
    config: TuningConfig,
    *,
    trials: int = 60,
    finalists: int = 5,
    seed: int = 0,
    storage: str | None = None,
    study_name: str | None = None,
    evaluator: Evaluator | None = None,
    verbose: bool = True,
) -> TuningResult:
    """Search ``policy``'s space over ``fold``, then choose on ``val_fold``.

    ``trials`` candidates are proposed by Optuna's TPE sampler and pruned by
    successive halving at the block granularity :attr:`TuningConfig.block` sets;
    the best ``finalists`` are then re-scored in full on ``val_fold`` and the
    winner of that is returned as :attr:`TuningResult.best_`.

    ``storage`` (a database URL) makes the study resumable under ``study_name``;
    without it the study lives in memory and dies with the process. ``evaluator``
    reuses an already-warm process pool, which is worth doing when a caller has one
    open -- otherwise one is built for this call and shut down after it.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    owned = evaluator is None
    evaluator = evaluator or Evaluator(config)
    try:
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=seed),
            pruner=SuccessiveHalvingPruner(min_resource=config.block),
            storage=storage,
            study_name=study_name,
            load_if_exists=storage is not None,
        )
        study.optimize(
            _objective(policy, fold, config, evaluator, verbose=verbose),
            n_trials=trials,
        )

        records = [_record(t) for t in study.trials if t.state != optuna.trial.TrialState.FAIL]
        chosen = _finalists(records, finalists)
        if verbose:
            print(
                f"re-scoring {len(chosen)} finalist(s) on {len(val_fold)} held-out "
                f"training economies"
            )
        for record in chosen:
            record["validation"] = score_params(
                policy, record["params"], val_fold, config, evaluator=evaluator
            )
            if verbose:
                print(
                    f"  {_params_str(record['params'])}  "
                    f"search {record['loss']:.3f} -> validation "
                    f"{record['validation']['loss']:.3f}"
                )
        best = min(chosen, key=lambda r: r["validation"]["loss"])
        return TuningResult(
            policy=policy.name,
            space=policy.space.describe(),
            config=config,
            trials_=records,
            finalists_=chosen,
            best_={
                "params": best["params"],
                "trial": best["number"],
                "search": {"loss": best["loss"], "groups": best["groups"]},
                "validation": best["validation"],
            },
        )
    finally:
        if owned:
            evaluator.close()


# -- internals -------------------------------------------------------------


def _objective(
    policy: TunablePolicy,
    fold: Fold,
    config: TuningConfig,
    evaluator: Evaluator,
    *,
    verbose: bool,
):
    """The trial body: walk the sequence in blocks, reporting as it goes.

    The running mean is reported after every block and the trial abandoned as soon
    as the pruner says so, which is why the per-group returns are accumulated here
    rather than left to :meth:`~control.tuning.objective.Evaluator.score`: a
    pruned trial's loss must be the loss over exactly the prefix it walked.
    """

    def objective(trial: optuna.Trial) -> float:
        params = policy.space.suggest(trial)
        prepared = policy.prepare(params, fold, config)
        returns: dict[int, float] = {}
        collapses: list[float] = []
        loss = float("inf")
        for block in fold.blocks(config.block):
            result = evaluator.score(policy, prepared, block)
            returns.update(result["per_group"])
            collapses.append(result["collapse_rate"])
            loss = -float(np.mean(list(returns.values())))
            trial.set_user_attr("loss", loss)
            trial.set_user_attr("groups", len(returns))
            trial.set_user_attr("collapse_rate", float(np.mean(collapses)))
            trial.set_user_attr("per_group", {str(k): v for k, v in returns.items()})
            trial.report(loss, step=len(returns))
            if trial.should_prune():
                if verbose:
                    print(
                        f"trial {trial.number:3d}  {_params_str(params)}  "
                        f"loss {loss:8.3f}  pruned at {len(returns)} economies"
                    )
                raise optuna.TrialPruned()
        if verbose:
            print(
                f"trial {trial.number:3d}  {_params_str(params)}  "
                f"loss {loss:8.3f}  over {len(returns)} economies"
            )
        return loss

    return objective


def _record(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    """One trial as a plain dict, whether it completed or was pruned.

    A pruned trial has no return value, so the loss is read from the user attrs the
    trial body writes after every block -- which is the loss over the prefix it
    actually walked, and the only honest number for it.
    """
    return {
        "number": trial.number,
        "params": dict(trial.params),
        "state": trial.state.name,
        "loss": float(trial.user_attrs.get("loss", np.inf)),
        "groups": int(trial.user_attrs.get("groups", 0)),
        "collapse_rate": float(trial.user_attrs.get("collapse_rate", 0.0)),
    }


def _finalists(records: Sequence[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    """The ``k`` candidates worth re-scoring, deepest-walked first.

    Trials that survived the whole sequence are preferred outright: their loss is
    an average over every economy in it, while a pruned trial's is an average over
    a prefix and not comparable to it. Pruned trials top the list up only when too
    few completed, ranked among themselves by loss, so a study whose pruner was
    harsh still has finalists to choose between.
    """
    if k < 1:
        raise ValueError(f"need at least one finalist, got {k}")
    complete = sorted(
        (r for r in records if r["state"] == "COMPLETE"), key=lambda r: r["loss"]
    )
    pruned = sorted(
        (r for r in records if r["state"] != "COMPLETE"),
        key=lambda r: (-r["groups"], r["loss"]),
    )
    chosen = (complete + pruned)[:k]
    if not chosen:
        raise RuntimeError("the study produced no usable trials")
    return chosen


def _params_str(params: Mapping[str, Any]) -> str:
    """Hyperparameters on one line, for a progress log."""
    return " ".join(
        f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
        for k, v in params.items()
    )
