"""What the policy sees: a stationary, standardised view of the economy.

Levels trend -- ``Y``, ``Yk``, ``P``, ``Vk`` and ``GD`` grow exponentially -- so a
network fed raw levels sees a distribution that walks away from it over an
episode and differs again between histories. The proxies already solve this for
their own fit with a
:class:`~economic_models.proxy.transform.StationarizingTransform`; the policy
reuses exactly that view, so agent and world model speak the same feature space::

    obs = state features (16: dlog(Y), dlog(Yk), PI, ER, dlog(P), dlog(Ck),
                              dlog(Ik), Rb, Rl, Rm, Rbl, GD/Y, PSBR/Y,
                              dlog(Vk), Q, PE)
        | exog features  (10: GRg, theta, dlog(Nfe), GRpr, ADDbl, Rln, NPLk,
                              Rbbar, NCAR, ro)
        | encoder latent (0 by default; ``encoder.latent_dim`` otherwise)

The exogenous block carries the actions applied in the period just observed, so
the policy sees where it left the levers without a separate memory.

**The latent block is the memory.** The two feature blocks above are a function of
the last two level rows alone, which makes the observation Markov in a world it is
not: the excitation drives hidden structural parameters (``RA``, ``gamma0``,
``eta0``, ``alpha1``, ``lambda40``) that the agent never sees and that persist for
many periods, and the observables are a projection of a larger structural state.
Appending a :class:`~economic_models.encoders.base.StateEncoder`'s filtered latent
gives the policy a recursive summary of the whole episode so far, so it can tell
which regime it is in rather than averaging over all of them.

That latent is carried as an **explicit belief**, passed into
:meth:`Observer.observe` and returned from it, never stored on the observer. Two
things depend on it: the environment can draw several transitions from one
position without their beliefs contaminating each other (see
:meth:`~control.env.CentralBankEnv.branch_step`), and an observation stays a pure
function of arguments, which is what lets the *same* observer serve a proxy
rollout and a ground-truth rollout.

The encoder is fit once on the history and then **frozen**: :meth:`Observer.fit`
learns it, and nothing afterwards updates it. It is a filter, not a trained part
of the agent -- ``advance`` reads the fitted parameters and returns a new belief
without touching them -- so the observation stays one fixed function of the
trajectory for the whole of training. Standardisation follows the same rule:
**fixed** statistics fit once on the history, never running ones. Proxy-side and
truth-side observations must be the same function of the same state, or a policy
cannot be transferred between them.

**Every episode starts from the same belief, and it is the whole history's.**
Filtering only a short tail at each reset would start the memory at the encoder's
*prior* -- for the Kalman encoder, its estimate of the latent at the beginning of
the history -- and leave the policy several steps of an episode reading a regime
signal that has not caught up yet. Since every episode branches from the end of
the run the observer was fit on, that belief is a constant: :meth:`Observer.fit`
filters the whole history once and keeps the posterior as
:attr:`Observer.branch_belief_`, and a reset simply reads it. Correct by
construction and free per episode, rather than approximated from a window.

The default encoder is the :class:`~economic_models.encoders.base.NullEncoder`,
whose latent is zero-width -- so the default observation is exactly the two
feature blocks, and a memoryless agent is the ``encoder=None`` special case rather
than a separate code path.
"""

from __future__ import annotations

from typing import Any, Self

import numpy as np

from economic_models.encoders import NullEncoder, StateEncoder
from economic_models.ground_truth import Run
from economic_models.interface import ModelInterface
from economic_models.proxy import StationarizingTransform


class Observer:
    """Builds and standardises the observation vector from level rows.

    Stateless with respect to an episode: the encoder belief is passed in and
    handed back (:meth:`observe`) rather than held, so every call is a pure
    function of its arguments. Fit once (:meth:`fit`) to learn the encoder and
    freeze the standardisation statistics.
    """

    def __init__(
        self,
        interface: ModelInterface,
        transform: StationarizingTransform | None = None,
        encoder: StateEncoder | None = None,
        clip: float = 10.0,
    ) -> None:
        """Wire the observer to the model ``interface`` the agent acts in.

        ``transform`` is the stationary feature view (defaults to one built from
        ``interface.transform_spec``, i.e. the same view the proxy is fit in);
        ``encoder`` supplies the filtered latent appended to each observation
        (defaults to a :class:`~economic_models.encoders.base.NullEncoder`, whose
        zero-width latent leaves the observation memoryless); ``clip`` bounds a
        standardised feature in absolute value, so one excursion cannot saturate
        the network.

        The encoder is this observer's own -- deliberately not the fitted proxy's,
        even though both are fit on the same history. The observation must be
        computable on the ground-truth side, where no proxy exists, so it cannot
        borrow a component that lives on only one side of the comparison.
        """
        self._interface = interface
        self._transform = transform or StationarizingTransform(interface.transform_spec)
        self._encoder = encoder or NullEncoder()
        self.clip = clip
        self.mean_: np.ndarray | None = None  # (dim,) fit on the history
        self.std_: np.ndarray | None = None  # (dim,)
        #: the belief every episode starts from: the whole history filtered up to
        #: (but not including) its last period, which the episode's first
        #: :meth:`observe` folds in. Handed out directly rather than copied -- the
        #: :meth:`~economic_models.encoders.base.StateEncoder.advance` purity
        #: contract means no caller can disturb it.
        self.branch_belief_: Any = None

    # -- shape ---------------------------------------------------------------

    @property
    def encoder(self) -> StateEncoder:
        """The state encoder whose latent this observer appends."""
        return self._encoder

    @property
    def names(self) -> tuple[str, ...]:
        """Human-readable names aligned with the observation columns.

        The latent block comes **last**, so a caller indexing an economic feature
        by name (:meth:`index`) is unaffected by which encoder is in use.
        """
        return (
            *self._transform.state_feature_names,
            *self._transform.exog_feature_names,
            *(f"z{j}" for j in range(self._encoder.latent_dim)),
        )

    @property
    def dim(self) -> int:
        """Width of the observation vector."""
        return len(self.names)

    @property
    def required_window(self) -> int:
        """Trailing level rows :meth:`init_belief` needs to warm-start.

        ``min_window`` feature rows warm-start the belief and one more is consumed
        by the first :meth:`observe`, which folds the branch period in itself so
        that reset and step advance the belief by exactly one period each; a
        feature row costs two level rows.

        This is the encoder's bare **minimum**, not what an episode should start
        from: a filter given its minimum starts at its prior and takes several
        periods to catch up. Episodes read :attr:`branch_belief_` instead, which
        is the same recursion run over the whole history.
        """
        return self._encoder.min_window + 2

    def index(self, name: str) -> int:
        """Column of the feature called ``name`` (one of :attr:`names`)."""
        return self.names.index(name)

    # -- fitting -------------------------------------------------------------

    def fit(self, history: Run) -> Self:
        """Fit the encoder on the historic run and freeze the standardisation.

        Both are learned here and neither moves again: the encoder becomes a fixed
        filter and the statistics fixed constants, so the observation is one
        unchanging function of the trajectory for the whole of training.

        The episode-start belief (:attr:`branch_belief_`) is filtered here too,
        off the same pass: every episode branches from this run's end, so the
        belief it starts from is a constant and there is nothing to recompute at
        reset. It stops one period short of that end for the reason
        :meth:`init_belief` does -- the episode's first :meth:`observe` folds the
        branch period in -- which lands it exactly on the last row
        :meth:`run_features` produces.
        """
        F, U = self._transform.transform_run(history)
        self._encoder.fit([(F, U)])
        features = np.hstack([F, U, self._encoder.encode_run(F, U)])
        std = features.std(axis=0)
        self.mean_ = features.mean(axis=0)
        # A constant column (std 0) standardises to 0 rather than exploding.
        self.std_ = np.where(std > 1e-12, std, 1.0)
        self.branch_belief_ = self._encoder.init_belief(F[:-1], U[:-1])
        return self

    def run_features(self, run: Run) -> np.ndarray:
        """Raw (unstandardised) observation rows for a whole run.

        The batch twin of :meth:`observe`: row ``i`` carries the causal latent
        summarising the run through period ``i``, which is the same latent the
        online path holds after folding period ``i`` in. Requires a fitted
        encoder.
        """
        F, U = self._transform.transform_run(run)
        return np.hstack([F, U, self._encoder.encode_run(F, U)])

    # -- online --------------------------------------------------------------

    def init_belief(self, window: tuple[np.ndarray, np.ndarray, np.ndarray]) -> Any:
        """Warm-start a belief from a trailing ``(states, params, actions)`` window.

        The window must hold at least :attr:`required_window` level rows. Its
        **last** period is deliberately left out: the first :meth:`observe` of an
        episode folds that period in, so warm start and step advance the belief by
        one period each and the branch period is counted exactly once.

        The general form, for a start the observer was not fit up to; an episode
        branching from the history's end reads :attr:`branch_belief_`, which is
        this run over the whole history. A filter given a short window starts at
        its prior, so the shorter the window the longer the belief takes to mean
        anything.
        """
        states, params, actions = window
        if len(states) < self.required_window:
            raise ValueError(
                f"need at least {self.required_window} level rows to warm-start, "
                f"got {len(states)}"
            )
        F = self._transform.transform_states(states)
        U = self._transform.transform_exog(params, actions)
        return self._encoder.init_belief(F[:-1], U[:-1])

    def observe(
        self, states: np.ndarray, exog: np.ndarray, belief: Any
    ) -> tuple[np.ndarray, Any]:
        """The standardised observation, and the belief that produced its latent.

        ``states`` is at least ``(2, n_state)`` of state levels and ``exog`` at
        least ``(2, n_exog)`` of parameter-then-action levels; only the final two
        rows of each are read. ``belief`` is the belief *before* this period: it is
        advanced by the realised row here, so the returned observation carries the
        latent summarising history **through** this period, matching row-for-row
        what :meth:`run_features` produces in batch.

        The input belief is not mutated (the
        :meth:`~economic_models.encoders.base.StateEncoder.advance` purity
        contract), so a caller may draw several transitions from one position and
        keep only the belief of the draw it commits.

        Raises :class:`ValueError` (from the transform) if a log-differenced level
        has gone non-positive -- the caller treats that as an economic collapse.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("observer must be fit before it can standardise")
        state_feat = self._transform.transform_states(states[-2:])[0]
        exog_feat = self._transform.transform_exog_row(exog[-1], exog[-2])
        belief = self._encoder.advance(belief, state_feat, exog_feat)
        raw = np.hstack([state_feat, exog_feat, self._encoder.latent(belief)])
        obs = np.clip((raw - self.mean_) / self.std_, -self.clip, self.clip)
        return obs, belief

    def unstandardise(self, obs: np.ndarray) -> np.ndarray:
        """The raw features behind a standardised observation row.

        The inverse of the standardisation in :meth:`observe` up to the clip -- a
        feature that saturated it comes back at the clip, not at its original
        value -- which lets a rule written in economic units (a Taylor rule, say)
        read an observation without being handed the state separately.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("observer must be fit before it can standardise")
        return np.asarray(obs, dtype=float) * self.std_ + self.mean_
