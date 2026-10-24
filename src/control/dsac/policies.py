"""The policies a run is scored with: the agent and the references it must beat.

A learned policy means nothing on its own -- its return has to be read against
what doing nothing clever earns in the same futures. Three references are kept
here beside the agent's own: uniform-random steps, the book's calibration held
fixed, and a Taylor rule on the bill rate. The Taylor rule is the bar that
matters, being *reactive* rather than constant, and it is the only one with
within-episode memory, which is why :func:`reset_policy` exists.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace

import numpy as np

from control.dsac.agent import DSACAgent
from control.dsac.config import TrainConfig
from control.env import CentralBankEnv
from control.observation import Observer
from economic_models.ground_truth.registry import DEFAULT_MODEL, ground_truth

#: A policy maps one observation to a normalised action in ``[-1, 1]``: a
#: fraction of the period's maximum lever move (see
#: :attr:`~control.env.EnvConfig.delta_rate`). Most are stateless functions of
#: the observation; one that carries within-episode memory (the Taylor rule's
#: smoothed growth gap) additionally offers a ``reset()``, which
#: :func:`reset_policy` calls at every episode boundary.
Policy = Callable[[np.ndarray], np.ndarray]


# -- policies ---------------------------------------------------------------


def reset_policy(policy: Policy) -> None:
    """Clear a policy's within-episode memory at an episode boundary.

    A no-op for the stateless policies, which is all of them but the Taylor rule
    -- rather than making every policy carry an empty ``reset``, this asks for one
    and does nothing when there is none, so a plain function stays a valid
    :data:`Policy`. Called by :func:`~control.dsac.evaluation.evaluate` and
    :func:`~control.dsac.evaluation.rollout` right after
    every :meth:`~control.env.CentralBankEnv.reset`.
    """
    reset = getattr(policy, "reset", None)
    if reset is not None:
        reset()


def free_instrument(config: TrainConfig) -> TrainConfig:
    """The run's configuration with an instrument too fast to constrain.

    One period's maximum move spans the whole box (``delta_rate = 1/dt``, so the
    step is exactly two normalised units), which makes
    :meth:`~control.env.CentralBankEnv.toward` reach any target *within* the
    period: a rule stated in levels behaves as its textbook self. The references
    are evaluated under this deliberately: the instrument's speed is part of the
    **agent's** policy problem, and a reference slowed to the agent's instrument
    would be a different rule than the one in the book -- note that this makes
    Taylor a *weaker* bar at sub-quarterly ``dt``, where a slower instrument
    damps the high-gain oscillation :func:`taylor_policy` documents. The agent's
    own environments keep the configured speed.
    """
    return replace(config, delta_rate=1.0 / config.dt)


def random_policy(action_dim: int, rng: np.random.Generator) -> Policy:
    """A policy that samples steps uniformly: a random walk over the box."""
    return lambda obs: rng.uniform(-1.0, 1.0, size=action_dim)


def constant_policy(env: CentralBankEnv, levels: Mapping[str, float]) -> Policy:
    """A policy that holds the levers at the given action *levels* (not normalised).

    "Holds" means steers, under the delta instrument
    (:meth:`~control.env.CentralBankEnv.toward`): the economy hands over
    wherever the history left the levers, so the baseline first travels to its
    own setting at the speed of the environment it is bound to and sits still
    thereafter. Bound to a :func:`free_instrument` environment -- which is how
    the references are scored -- it arrives within the first period.
    """
    levels = dict(levels)
    return lambda obs: env.toward(levels)


def taylor_policy(
    env: CentralBankEnv,
    observer: Observer,
    *,
    dt: float,
    pi_target: float = 0.02,
    phi_pi: float = 0.5,
    phi_y: float = 0.5,
    model: str = DEFAULT_MODEL,
) -> Policy:
    """The textbook Taylor rule on the bill rate, calibration on the other levers.

    ``lever = i* + (own + phi_pi)(PI - pi_target) + phi_y * growth_gap``, where the
    lever and the response ``own`` it already owes come from the model's spec:
    for GROWTH the lever is the bill rate and ``own`` is one, so this is the
    textbook rule; for Smets-Wouters the lever is the *deviation* from a policy
    rule already inside the model, and ``own`` is zero because that rule already
    responds to inflation -- a reference there adds to what the estimated Federal
    Reserve was doing rather than replacing it. In GROWTH's case,
    ``i*`` is the calibration bill rate -- so at target inflation and at potential
    growth the rule *is* the calibration baseline, and every difference between the
    two references is the rule reacting to the cycle. The response to inflation is
    ``1 + phi_pi`` because the nominal rate must move with ``PI`` one-for-one
    before it moves the *real* rate at all (the Taylor principle); the defaults
    ``phi_pi = phi_y = 0.5`` are Taylor's original pair.

    The gap term is a **growth** gap, not a level output gap: the same
    ``realised - potential`` annualised growth the mandate scores (see
    :class:`~control.rewards.mandate.MandateReward`), because potential *output*
    is not identified from an observation whereas potential *growth* is
    (``GRpr`` plus full-employment labour growth). ``NCAR`` and ``ro`` are held at
    calibration: this is a monetary rule, and leaving the other two levers fixed
    is what makes it comparable to the baseline.

    The gap is **smoothed over roughly a year** before it is acted on, by the same
    exponential recursion the model itself uses to build ``PI`` from the one-period
    price change (11.10: ``x = dt*x_raw + (1 - dt)*x(-1)``). Without it the rule is
    not frequency-invariant and destabilises the economy at sub-quarterly ``dt``:
    ``dlog(Yk)/dt`` annualises a *single* period's log-difference, whose transitory
    component scales as ``sqrt(dt)`` rather than ``dt``, so the effective feedback
    gain on it grows as ``1/sqrt(dt)`` while the transmission lag stays fixed in
    calendar time -- high gain against an unchanged delay, which shows up as a
    period-2 flip oscillation that the action clip then sustains as a limit cycle.
    The smoother's gain at that flip frequency is ``dt/(2 - dt)``, which cancels the
    ``1/dt`` annualisation almost exactly, so ``phi_y`` means the same thing at
    quarterly and monthly steps. ``PI`` needs no such treatment: it arrives already
    smoothed this way, which is why only the growth leg had to be fixed.

    Carrying that smoothed gap makes the rule **stateful**: it holds one scalar
    between steps and :meth:`reset` clears it at an episode boundary, which
    :func:`reset_policy` does for every roll. It starts each episode at zero -- the
    branch point sits at the book's calibration, which grows at potential -- so the
    rule opens at the calibration rate and earns its deviations from data.

    The rule reads the standardised observation and inverts it, so it consumes
    exactly what the agent does and can be scored on the same futures. It drives
    a delta instrument (:meth:`~control.env.CentralBankEnv.toward`): the rate it
    computes is a *target*, travelled toward at the speed of the environment it
    is bound to -- conventionally a :func:`free_instrument` one, under which the
    move completes within the period and the rule is its textbook self, exempt
    by design from the speed that constrains the agent. Targets outside
    :attr:`~control.env.EnvConfig.action_bounds` clip to the box, and that lower
    bound **binds often**: the book's calibration settles with inflation around
    1%, so a rule aiming at 2% cuts into the floor and spends much of an episode
    there. The rule is therefore a reference, not a well-tuned policy -- which is
    the point of scoring it next to the constant baseline rather than instead of
    it.
    """
    return _TaylorRule(
        env,
        observer,
        dt=dt,
        pi_target=pi_target,
        phi_pi=phi_pi,
        phi_y=phi_y,
        model=model,
    )


class _TaylorRule:
    """The stateful body of :func:`taylor_policy`; see it for the rule itself.

    A class rather than a closure because the rule carries the smoothed growth gap
    between steps and has to be told where an episode ends -- state a plain
    function would have to hide in a cell and expose by attribute.
    """

    def __init__(
        self,
        env: CentralBankEnv,
        observer: Observer,
        *,
        dt: float,
        pi_target: float,
        phi_pi: float,
        phi_y: float,
        model: str = DEFAULT_MODEL,
    ) -> None:
        """Bind the rule to an environment and resolve its feature columns."""
        spec = ground_truth(model)
        self._env = env
        self._observer = observer
        self._dt = dt
        self._pi_target = pi_target
        self._phi_pi = phi_pi
        self._phi_y = phi_y
        self._levels = dict(calibration_actions(model))
        self._lever = spec.reference_lever
        self._i_star = self._levels[self._lever]
        self._own_response = spec.reference_inflation_response
        self._pi_at, self._growth_at, self._potential_at, self._labour_at = (
            observer.index(name) for name in ("PI", "dlog(Yk)", "GRpr", "dlog(Nfe)")
        )
        self._gap = 0.0

    def reset(self) -> None:
        """Clear the smoothed growth gap; the next episode opens at potential."""
        self._gap = 0.0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """The bill rate the rule sets, given one standardised observation."""
        raw = self._observer.unstandardise(obs)
        dt = self._dt
        gap = raw[self._growth_at] / dt - (
            raw[self._potential_at] + raw[self._labour_at] / dt
        )
        # 11.10's smoother, applied to the growth gap: a ~1-year time constant at
        # any dt, which is what keeps phi_y frequency-invariant.
        self._gap = dt * gap + (1.0 - dt) * self._gap
        rate = (
            self._i_star
            + (self._own_response + self._phi_pi) * (raw[self._pi_at] - self._pi_target)
            + self._phi_y * self._gap
        )
        return self._env.toward({**self._levels, self._lever: rate})


def agent_policy(agent: DSACAgent, *, deterministic: bool = True) -> Policy:
    """A policy that queries the agent's actor."""
    return lambda obs: agent.act(obs, deterministic=deterministic)


def calibration_actions(model: str = DEFAULT_MODEL) -> dict[str, float]:
    """Where ``model``'s levers rest when the bank does nothing.

    The reference a policy has to beat. For GROWTH that is the book's calibrated
    settings; for Smets-Wouters it is the zero deviation from the estimated policy
    rule, which is to say the Federal Reserve's own estimated reaction function.
    """
    return ground_truth(model).baseline_actions()
