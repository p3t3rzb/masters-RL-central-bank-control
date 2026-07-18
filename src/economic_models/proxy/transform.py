"""The run feature transform: the stationary view a proxy is fit on.

GROWTH-model levels trend, so a regression on them is ill-posed; every proxy is
fit on this stationary feature view instead. The trending columns are
stationarized before a proxy sees them:

* trending aggregates (``Y``, ``Yk``, ``P``, ``Ck``, ``Ik``, ``Vk``) enter as
  one-step log-differences (growth rates per step);
* government stocks/flows (``GD``, ``PSBR``) enter as ratios to nominal GDP;
* everything already stationary (inflation, rates, ratios) enters as its level.

State features are invertible given the previous step's levels
(:meth:`~StationarizingTransform.transform_states` /
:meth:`~StationarizingTransform.invert_states`), so a proxy can predict them and
roll out in level space. Exogenous inputs are always given, never predicted, so
they are forward-only (``Nfe`` log-differenced; the rest pass through as levels).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from economic_models.variables import Actions, Parameters, State

if TYPE_CHECKING:
    from economic_models.ground_truth import Run

#: state columns entered as one-step log-differences
LOG_DIFF = ("Y", "Yk", "P", "Ck", "Ik", "Vk")
#: state columns entered as a ratio to nominal GDP (``Y``)
RATIO_TO_Y = ("GD", "PSBR")
#: exogenous columns entered as one-step log-differences
EXOG_LOG_DIFF = ("Nfe",)


class StationarizingTransform:
    """Map a run between level space and a stationary feature space.

    Stationarizes trending levels via one-step log-differences and GDP ratios;
    invertible given the previous step's levels. The shared input-variable names
    are fixed here.
    """

    #: endogenous input variables, aligned with :class:`State` columns
    state_names: tuple[str, ...] = State.names()
    #: exogenous input variables, aligned with ``Parameters`` then ``Actions``
    exog_names: tuple[str, ...] = (*Parameters.names(), *Actions.names())

    def __init__(self) -> None:
        """Precompute the per-column log-diff / ratio masks and feature names."""
        state = self.state_names
        self._state_log_diff = np.array([name in LOG_DIFF for name in state])
        self._state_ratio = np.array([name in RATIO_TO_Y for name in state])
        self._y_col = state.index("Y")
        # Names of the log-diffed columns, in mask order -- used to name the
        # offending series if a non-positive level ever reaches the log.
        self._state_log_diff_names = tuple(n for n in state if n in LOG_DIFF)
        self._state_feature_names = tuple(
            f"dlog({name})" if name in LOG_DIFF
            else f"{name}/Y" if name in RATIO_TO_Y
            else name
            for name in state
        )

        exog = self.exog_names
        self._exog_log_diff = np.array([name in EXOG_LOG_DIFF for name in exog])
        self._exog_log_diff_names = tuple(n for n in exog if n in EXOG_LOG_DIFF)
        self._exog_feature_names = tuple(
            f"dlog({name})" if name in EXOG_LOG_DIFF else name for name in exog
        )

    @staticmethod
    def _log(block: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
        """``np.log`` after checking every masked column is strictly positive.

        Log-differencing a trending level assumes it stays positive; a slump can
        drive a series (e.g. ``Ik``) non-positive, which would otherwise yield a
        silent ``nan`` feature that poisons the whole fit. Raise instead, naming
        the offending series, so the failure is loud and localised.
        """
        if not np.all(block > 0):
            bad = [n for j, n in enumerate(names) if not np.all(block[..., j] > 0)]
            raise ValueError(
                f"cannot log-difference non-positive level(s) {bad}; the "
                "stationarizing transform assumes these series stay strictly positive"
            )
        return np.log(block)

    # -- feature naming ------------------------------------------------------

    @property
    def state_feature_names(self) -> tuple[str, ...]:
        """Human-readable names aligned with the state feature columns."""
        return self._state_feature_names

    @property
    def exog_feature_names(self) -> tuple[str, ...]:
        """Human-readable names aligned with the exogenous feature columns."""
        return self._exog_feature_names

    # -- endogenous side (invertible) ---------------------------------------

    def transform_states(self, states: np.ndarray) -> np.ndarray:
        """State features for rows ``1..T-1`` of a ``(T, n_state)`` level array.

        Row ``i`` of the output is the feature built from level rows ``i`` and
        ``i + 1``, so the output has ``T - 1`` rows.
        """
        current, previous = states[1:], states[:-1]
        features = current.copy()
        names = self._state_log_diff_names
        features[:, self._state_log_diff] = self._log(
            current[:, self._state_log_diff], names
        ) - self._log(previous[:, self._state_log_diff], names)
        features[:, self._state_ratio] = (
            current[:, self._state_ratio] / current[:, [self._y_col]]
        )
        return features

    def invert_states(
        self, features: np.ndarray, previous_levels: np.ndarray
    ) -> np.ndarray:
        """Recover one row of levels from one feature row and the prior levels.

        The inverse of one row of :meth:`transform_states`; enables open-loop
        rollout in level space.
        """
        levels = features.copy()
        levels[self._state_log_diff] = previous_levels[self._state_log_diff] * np.exp(
            features[self._state_log_diff]
        )
        levels[self._state_ratio] = features[self._state_ratio] * levels[self._y_col]
        return levels

    # -- exogenous side (forward only) --------------------------------------

    def transform_exog(self, params: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Exogenous features for rows ``1..T-1`` of aligned ``(T, .)`` arrays."""
        exog = np.hstack([params, actions])
        features = exog[1:].copy()
        names = self._exog_log_diff_names
        features[:, self._exog_log_diff] = self._log(
            exog[1:, self._exog_log_diff], names
        ) - self._log(exog[:-1, self._exog_log_diff], names)
        return features

    def transform_exog_row(
        self, exog_now: np.ndarray, exog_prev: np.ndarray
    ) -> np.ndarray:
        """Single exogenous feature row from the current and previous levels.

        The one-step form of :meth:`transform_exog`, used when a rollout feeds
        exogenous inputs in one period at a time. ``exog_now`` / ``exog_prev``
        are level rows ordered like :attr:`exog_names`.
        """
        features = exog_now.astype(float).copy()
        mask = self._exog_log_diff
        names = self._exog_log_diff_names
        features[mask] = self._log(exog_now[mask], names) - self._log(
            exog_prev[mask], names
        )
        return features

    # -- convenience ---------------------------------------------------------

    def transform_run(self, run: Run) -> tuple[np.ndarray, np.ndarray]:
        """``(state_features, exog_features)`` for a whole :class:`Run`."""
        return (
            self.transform_states(run.states),
            self.transform_exog(run.params, run.actions),
        )
