"""What a tunable policy is: a search space, and how to build one from a draw.

Two policies could hardly be less alike in what "setting the hyperparameters"
costs. A Taylor rule is *free* to build -- two gains, a constructor call, no
fitting -- so a candidate is a dict and the whole cost of a trial is the rolling
out. A DSAC agent is the opposite: its hyperparameters are the terms of a training
run, so a candidate costs a training run, paid once and then reused across every
economy the fold contains.

:class:`TunablePolicy` is the template that holds both. It splits building into
three hooks that fire at three different rates:

* :meth:`~TunablePolicy.prepare` -- once per candidate, before any economy is
  touched. Where an expensive candidate is actually built.
* :meth:`~TunablePolicy.observer` -- once per economy. What the policy sees, which
  a fitted policy must be allowed to bring with it rather than have re-derived.
* :meth:`~TunablePolicy.build` -- once per economy, cheap by construction: bind
  the prepared candidate to this environment.

Each hook is handed the material at its own rate: ``prepare`` gets the fold, so it
can load and learn from every economy in it; ``observer`` and ``build`` get the
:class:`~control.world.TrainingWorld`, whose
:attr:`~control.world.TrainingWorld.history` is that group's simulated main run --
the only data a proxy or an observation is ever fit on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Mapping

from economic_models.ground_truth import GROWTH_INTERFACE

from control.dsac.train import Policy, taylor_policy
from control.env import CentralBankEnv
from control.observation import Observer
from control.tuning.space import Real, SearchSpace
from control.world import TrainingWorld

if TYPE_CHECKING:
    from control.tuning.objective import Fold, TuningConfig


# -- the template ----------------------------------------------------------


class TunablePolicy(ABC):
    """A policy family, its hyperparameters, and how a draw of them is realised."""

    #: the name this family is selected by on the command line.
    name: ClassVar[str] = "policy"

    # -- what may be varied --------------------------------------------------

    @property
    @abstractmethod
    def space(self) -> SearchSpace:
        """The hyperparameters a search may vary, and their ranges."""

    @abstractmethod
    def defaults(self) -> dict[str, Any]:
        """The untuned settings: the reference a tuned draw has to beat.

        Every family here has a conventional configuration -- Taylor's original
        pair of gains, a published set of agent hyperparameters -- and reporting
        the tuned result without it says nothing about whether the search helped.
        """

    # -- realising a draw ----------------------------------------------------

    def prepare(
        self, params: Mapping[str, Any], fold: "Fold", config: "TuningConfig"
    ) -> Any:
        """Turn one draw of the hyperparameters into whatever :meth:`build` needs.

        Called once per candidate, before any economy is scored, and handed the
        ``fold`` it will be scored on so that a family which must *learn* something
        first can learn it from those economies' histories. The returned handle is
        opaque to the search and travels to the workers, so it has to be picklable.

        The default is the cheap case: the parameters themselves, unchanged.
        """
        return dict(params)

    @abstractmethod
    def build(
        self,
        prepared: Any,
        env: CentralBankEnv,
        observer: Observer,
        world: TrainingWorld,
        *,
        dt: float,
    ) -> Policy:
        """Bind a prepared candidate to one economy's environment.

        Called once per group and expected to be cheap: the expensive half of
        constructing a candidate belongs in :meth:`prepare`. ``world`` is passed as
        well as ``env`` because a family may want the group's history here and not
        only its environment.
        """

    def observer(self, prepared: Any, world: TrainingWorld) -> Observer:
        """What the policy sees in this economy.

        The default fits a fresh, memoryless observer on the group's own history,
        which is right for a rule stated in economic units: such a rule reads
        :meth:`~control.observation.Observer.unstandardise`, which inverts each
        column on its own, so the economic features come back identically whatever
        latent block an encoder would have appended.

        A *fitted* policy must override this and return the observer it was
        trained with -- standardisation statistics are part of a trained agent, and
        re-deriving them per economy would move the agent's inputs under it. That
        is why ``prepared`` is passed: it is where such an observer would be kept.
        """
        return Observer(GROWTH_INTERFACE).fit(world.history)

    @property
    def free_instrument(self) -> bool:
        """Whether the levers move at textbook speed rather than the agent's.

        True for the reference rules, which are stated in levels and would be a
        different rule if slowed down; false for a policy whose instrument speed is
        part of the problem it was optimised against.
        """
        return True


# -- the Taylor rule -------------------------------------------------------


@dataclass(frozen=True)
class TaylorTuning(TunablePolicy):
    """The Taylor rule's two feedback gains.

    ``Rbbar = i* + (1 + phi_pi)(PI - pi_target) + phi_y * growth_gap``, of which
    the search varies ``phi_pi`` and ``phi_y``. The inflation target is *not*
    varied and is not meant to be: it is the target the mandate scores the rule
    against, and letting the search move it would tune the rule's objective rather
    than its response.

    ``pi_target`` is **not** searchable and is not a field a study should move: it
    is the target the mandate scores the rule against, and letting a search
    choose it would tune the objective rather than the response.

    ``phi_min`` and ``phi_max`` bound both gains, and the range **includes
    negative values** for a reason the rule's own algebra makes plain. The
    response to inflation is ``1 + phi_pi``, so:

    * ``phi_pi = -1`` switches the inflation leg off entirely, leaving
      ``rate = i* + phi_y * gap``;
    * ``phi_pi = -1`` together with ``phi_y = 0`` leaves ``rate = i*`` at every
      period, which -- since the other two levers are already held at calibration
      -- **is** the constant baseline, exactly and not approximately.

    A range starting at zero therefore cannot express the baseline the rule is
    compared against, and in this model that is not an abstract loss: the
    calibration settles near 0.6% inflation against a 2% target, so the smallest
    admissible response ``(1 + 0)`` puts the rate a persistent ~1.4 points below
    calibration before the rule reacts to anything. Bounding at ``-1.5`` puts the
    baseline in the *interior*, so a search landing near it has chosen it rather
    than been stopped there. Linear, never log: the range spans zero.
    """

    name: ClassVar[str] = "taylor"
    pi_target: float = 0.02  #: the inflation target, held at the mandate's
    phi_min: float = -1.5  #: lower bound on both gains; below -1 the rule is perverse
    phi_max: float = 1.5  #: upper bound on both gains

    @property
    def space(self) -> SearchSpace:
        """Both gains, linearly over ``[phi_min, phi_max]``."""
        return SearchSpace(
            [
                Real("phi_pi", self.phi_min, self.phi_max),
                Real("phi_y", self.phi_min, self.phi_max),
            ]
        )

    def defaults(self) -> dict[str, Any]:
        """Taylor's original pair, which every call site in this repository uses."""
        return {"phi_pi": 0.5, "phi_y": 0.5}

    def build(
        self,
        prepared: Any,
        env: CentralBankEnv,
        observer: Observer,
        world: TrainingWorld,
        *,
        dt: float,
    ) -> Policy:
        """The rule itself, bound to this environment and its observer."""
        return taylor_policy(
            env, observer, dt=dt, pi_target=self.pi_target, **prepared
        )


#: The policy families a search may be run over, by command-line name.
POLICIES: dict[str, type[TunablePolicy]] = {"taylor": TaylorTuning}
