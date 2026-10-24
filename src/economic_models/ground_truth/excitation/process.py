"""Model-agnostic excitation of exogenous inputs: the config and the per-step process.

A proxy can only learn the causal effect of the policy levers if the historic
run actually moved them. Every bank-visible exogenous input therefore drifts as a
clipped AR(1) around its baseline; the hidden structural parameters drift too,
reaching the dataset only as unexplained variance in the visible state. On top of
that drift sit optional stochastic-volatility, crisis and per-run-climate layers.

This module is the *what* of excitation -- the spec and the stateful draw -- kept
apart from :mod:`economic_models.ground_truth.excitation.generator`, the *how* of
driving a model under it, because a :class:`~economic_models.run.Scenario` needs
only the former: futures are drawn from the process alone, with no model solved.

* :class:`ExcitationConfig` -- the generic drift/volatility/crisis/climate bundle;
* :class:`ExcitationProcess` -- the per-step draw of exogenous inputs, with a
  model hook (:meth:`~ExcitationProcess._model_inputs`) for the model's own
  special inputs (e.g. a stabilizer or a level random walk).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from economic_models.ground_truth.excitation.specs import (
    AR1Spec,
    ClimateSpec,
    CrisisSpec,
    ExcitationJitter,
    StochasticVolatilitySpec,
)


@dataclass(frozen=True)
class ExcitationConfig:
    """The model-agnostic tunables of how a run's exogenous inputs are excited.

    ``visible`` and ``hidden`` map an input name to its :class:`AR1Spec`; the
    baselines the deviations are taken around come from the model calibration,
    not from here. A model's *special* inputs (a stabilizer, a level random walk)
    are handled by its :class:`ExcitationProcess` subclass, configured by fields
    a subclass of this config adds.
    """

    visible: Mapping[str, AR1Spec]  #: bank-visible Parameters/Actions drift
    hidden: Mapping[str, AR1Spec]  #: hidden structural parameter drift
    #: optional stochastic-volatility regime scaling every innovation; ``None``
    #: leaves innovation sizes constant (the calm default).
    volatility: StochasticVolatilitySpec | None = field(default=None, kw_only=True)
    #: optional rare recoverable crisis shocks; ``None`` disables them.
    crisis: CrisisSpec | None = field(default=None, kw_only=True)
    #: optional per-run turbulence draw mixing calm and crisis-prone runs;
    #: ``None`` gives every run the same character.
    climate: ClimateSpec | None = field(default=None, kw_only=True)

    @property
    def hidden_names(self) -> tuple[str, ...]:
        """Hidden parameter names, in the column order recorded on a :class:`Run`."""
        return tuple(self.hidden)

    def perturbed(
        self, rng: np.random.Generator, jitter: ExcitationJitter
    ) -> ExcitationConfig:
        """A neighbouring economy: the same variables, drifting differently.

        Every spec is redrawn around itself (see :class:`ExcitationJitter`), so
        the result excites the same inputs under the same corridor and records
        the same columns -- it is substitutable for this config anywhere, and in
        particular is a legal ``continuation_config`` -- while the dynamics an
        encoder filters and a proxy fits are no longer the ones they were fitted
        to.

        This is the difference between deploying onto *another draw* of the
        economy the offline stack was built in and deploying into an economy of
        comparable difficulty it has never seen. Only the second measures
        transfer, and only the second gives an online correction a systematic
        error to find rather than sampling noise to chase.
        """
        return replace(
            self,
            visible={n: s.perturbed(rng, jitter) for n, s in self.visible.items()},
            hidden={n: s.perturbed(rng, jitter) for n, s in self.hidden.items()},
            volatility=(
                None
                if self.volatility is None
                else self.volatility.perturbed(rng, jitter)
            ),
            crisis=None if self.crisis is None else self.crisis.perturbed(rng, jitter),
            climate=None
            if self.climate is None
            else self.climate.perturbed(rng, jitter),
        )._perturb_model_specs(rng, jitter)

    def _perturb_model_specs(
        self, rng: np.random.Generator, jitter: ExcitationJitter
    ) -> ExcitationConfig:
        """Perturb a subclass's own specs on top of :meth:`perturbed` (default: none).

        The generic half has already been done and handed over as ``self``; a
        subclass that adds specs of its own -- a stabilizer gain, a level walk --
        redraws them here and returns the result.
        """
        return self


class ExcitationProcess(ABC):
    """Stateful per-step draws of the exogenous inputs for one :class:`Run`.

    Owns the model-agnostic machinery -- the stochastic-volatility regime, the
    crisis episodes and the clipped-AR(1) drift of every ``visible``/``hidden``
    input. Each :meth:`step` advances all of that, then calls the abstract
    :meth:`_model_inputs` hook so the concrete model can add or override its own
    special inputs (a stabilizer, a level random walk, ...) using the ``feedback``
    signal the generator passes in.
    """

    def __init__(
        self,
        config: ExcitationConfig,
        baselines: Mapping[str, float],
        rng: np.random.Generator,
        climate: float | None = None,
        dt: float = 1.0,
    ) -> None:
        """Initialise the per-step draw state for one run.

        ``config`` is the excitation spec; ``baselines`` are the input levels
        deviations are taken around; ``rng`` is the run's random generator;
        ``climate`` in ``[0, 1]`` biases volatility and crisis frequency for the
        whole run (``None`` for no bias); ``dt`` is the period length in years
        used to rescale annual rates to per-step ones.
        """
        self._config = config
        self._baselines = baselines
        self._rng = rng
        self._dt = dt
        # Kept so :meth:`reconfigure` can re-derive the climate's effect under a
        # different config: the draw belongs to the run, the response to the spec.
        self._climate = climate
        self._ar1 = {**config.visible, **config.hidden}
        self._devs = {name: 0.0 for name in self._ar1}
        self._logvol = 0.0
        # Live crisis episodes as ``(deviation_by_input, per_step_decay)`` pairs,
        # summed each step into ``_crisis_total``; several may overlap, each fading
        # at its own drawn rate. ``_crisis_total`` is the level shock applied now.
        self._crisis_episodes: list[tuple[dict[str, float], float]] = []
        self._crisis_total: dict[str, float] = {}
        self._adopt(config)
        self._since_crisis = self._min_gap_steps
        #: diagnostics exposed after each :meth:`step` (not part of the interface)
        self.vol_multiplier = 1.0
        self.crisis_intensity = 0.0
        self._init_model_state()

    def reconfigure(
        self, config: ExcitationConfig, climate: float | None = None
    ) -> None:
        """Continue this process under a different spec, keeping where it is.

        Swaps the config and everything derived from it while leaving the
        *state* -- the AR(1) deviations, the volatility regime, any live crisis
        episodes and the model's own drift state -- exactly where it stands. A
        process reconfigured mid-run is therefore an economy whose weather
        changed, not an economy restarted in different weather: it is at the same
        point of the same drift, and only what happens next is drawn differently.

        That is what makes it a *structural break* rather than a second sample.
        Inputs the new config excites and the old one did not start at their
        baseline (deviation zero), which is the only sensible reading of "this
        variable was not moving before".

        ``climate`` replaces this run's turbulence draw, and is how a preset that
        has a :class:`~economic_models.ground_truth.excitation.specs.ClimateSpec`
        takes over from one that does not: with nothing passed the old draw
        stands, which for a calm history means ``None`` -- every future would
        then be neutral, and a bank of futures that all have the same character
        is exactly the heterogeneity the spec exists to provide.

        Used by :meth:`ExcitedRunGenerator.generate_with_continuations` to fork
        futures under a different preset than the history was drawn under -- see
        :attr:`~control.world.WorldConfig.deploy_excitation`.
        """
        if climate is not None:
            self._climate = climate
        old = dict(self._ar1)
        self._adopt(config)
        self._devs = {
            name: self._carried_deviation(name, old.get(name), spec)
            for name, spec in self._ar1.items()
        }
        self._since_crisis = min(self._since_crisis, self._min_gap_steps)
        self._reconfigure_model_state()

    def _carried_deviation(self, name: str, old: AR1Spec | None, new: AR1Spec) -> float:
        """One input's deviation, re-expressed against the new spec's own centre.

        A spec's deviation is measured from the level it reverts to, so when a
        reconfiguration moves that level the same number means a different
        economy. Re-basing keeps the **level** continuous across the branch: the
        input carries on from exactly where it was, and the AR(1) then pulls it
        toward its new resting place over the input's own persistence.

        That gradualness is the point. A structural break implemented as an
        instantaneous jump in a hidden parameter is a step change the proxy sees
        in one period and the excitation never produces; a break implemented as a
        moved steady state is a slow, systematic, *persistent* divergence between
        the fitted model and the world -- which is what a real regime change
        looks like and what the correction's forgetting factor and bias state
        were built to track.

        An input the previous spec did not excite starts at deviation zero, which
        is the only sensible reading of "this was not moving before".
        """
        dev = self._devs.get(name, 0.0)
        if old is None:
            return 0.0
        base = float(self._baselines[name])
        return dev + float(
            np.clip(base + old.center, old.lower, old.upper)
            - np.clip(base + new.center, new.lower, new.upper)
        )

    def _adopt(self, config: ExcitationConfig) -> None:
        """Take ``config`` and rebuild every cache derived from it."""
        self._config = config
        self._ar1 = {**config.visible, **config.hidden}
        # ``min_gap`` is in years; convert to steps for the per-step onset guard.
        self._min_gap_steps = (
            round(config.crisis.min_gap / self._dt) if config.crisis else 0
        )
        # This run's climate biases volatility and crisis frequency for its whole
        # life; a neutral 0.5-equivalent (no bias) applies when unset.
        if config.climate is not None and self._climate is not None:
            self._vol_offset = config.climate.vol_offset(self._climate)
            self._crisis_scale = config.climate.crisis_scale(self._climate)
        else:
            self._vol_offset = 0.0
            self._crisis_scale = 1.0

    # -- subclass contract -------------------------------------------------

    def _reconfigure_model_state(self) -> None:
        """Rebuild model-specific caches after :meth:`reconfigure` (default: none).

        Distinct from :meth:`_init_model_state`, which also *resets* the drift
        state a reconfiguration is required to preserve. A subclass that caches
        anything off the config -- a spec pulled out of it, say -- rebuilds it
        here and touches nothing else.
        """

    def _init_model_state(self) -> None:
        """Initialise any model-specific per-run state (default: none).

        Called at the end of ``__init__``; a subclass that carries extra draw
        state for its special inputs (an AR(1) gap, a log-level deviation, ...)
        seeds it here.
        """

    @abstractmethod
    def _model_inputs(self, values: dict[str, float], feedback: Any) -> None:
        """Fill in the model's special exogenous inputs, mutating ``values``.

        Called by :meth:`step` after the generic AR(1)/volatility/crisis draws
        have populated ``values`` with every ``visible``/``hidden`` input.
        ``feedback`` is the model-specific signal the generator reads off the
        last solved state (for GROWTH, the previous employment rate). Reads the
        live ``self.vol_multiplier`` for volatility scaling.
        """

    # -- generic drift/volatility/crisis machinery -------------------------

    def _advance_volatility(self) -> float:
        """Advance the stochastic-volatility regime and return this step's multiplier."""
        vol = self._config.volatility
        if vol is None:
            return 1.0
        self._logvol = vol.advance(self._logvol, self._rng, dt=self._dt)
        # Climate shifts where in the band this run sits, but the clip still caps
        # the total so turbulence never leaves the solver-safe corridor.
        total = float(
            np.clip(self._logvol + self._vol_offset, -vol.max_logvol, vol.max_logvol)
        )
        return vol.multiplier(total)

    def _advance_crisis(self) -> None:
        """Fade the live crisis episodes, then maybe erupt a fresh one."""
        crisis = self._config.crisis
        if crisis is None:
            return
        self._fade_crisis_episodes()
        self._since_crisis += 1
        self._maybe_erupt_crisis(crisis)

        total: dict[str, float] = {}
        for dev, _ in self._crisis_episodes:
            for name, value in dev.items():
                total[name] = total.get(name, 0.0) + value
        self._crisis_total = total
        self.crisis_intensity = sum(abs(v) for v in total.values())

    def _fade_crisis_episodes(self) -> None:
        """Fade every live episode by its own decay; drop the spent ones."""
        faded = []
        for dev, decay in self._crisis_episodes:
            dev = {name: value * decay for name, value in dev.items()}
            if sum(abs(v) for v in dev.values()) > 1e-6:
                faded.append((dev, decay))
        self._crisis_episodes = faded

    def _maybe_erupt_crisis(self, crisis: CrisisSpec) -> None:
        """Draw whether a fresh ``crisis`` episode erupts this step, and start it."""
        # ``prob`` is the annual onset probability; the per-step hazard that yields
        # the same annual rate is ``1 - (1-prob)**dt``.
        onset_prob = (1.0 - (1.0 - crisis.prob) ** self._dt) * self._crisis_scale
        if self._since_crisis < self._min_gap_steps or self._rng.random() >= onset_prob:
            return
        severity = self._rng.uniform(*crisis.severity_range)
        # ``decay_range`` is the annual decay; the per-step decay is its ``dt`` power.
        decay = self._rng.uniform(*crisis.decay_range) ** self._dt
        dev = {name: mag * severity for name, mag in crisis.impulses.items()}
        # A minority of crises are also financial: an extra impulse bundle on
        # top of the base one, drawn independently at onset.
        if crisis.financial_impulses and self._rng.random() < crisis.financial_prob:
            for name, mag in crisis.financial_impulses.items():
                dev[name] = dev.get(name, 0.0) + mag * severity
        self._crisis_episodes.append((dev, decay))
        self._since_crisis = 0

    def step(self, feedback: Any) -> dict[str, float]:
        """Draw this step's exogenous input levels given the model ``feedback``.

        Advances the volatility and crisis state, then every AR(1)/random-walk
        input, then defers to :meth:`_model_inputs` for the model's special
        inputs. ``feedback`` is the model-specific signal read off the last
        solved state (passed straight through to :meth:`_model_inputs`).
        """
        self.vol_multiplier = self._advance_volatility()
        self._advance_crisis()

        values: dict[str, float] = {}
        for name, spec in self._ar1.items():
            base = float(self._baselines[name])
            self._devs[name], values[name] = spec.advance(
                self._devs[name],
                base,
                self._rng,
                sigma_scale=self.vol_multiplier,
                extra=self._crisis_total.get(name, 0.0),
                dt=self._dt,
            )

        self._model_inputs(values, feedback)
        return values
