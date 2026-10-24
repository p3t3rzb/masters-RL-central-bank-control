"""The terms of one training run, as a single frozen value.

A run is defined by four groups of choices -- the world it is set in, the world
model standing in for it, the mandate it is scored against, and the agent's own
optimisation knobs -- and every one of them is a field here. Keeping them in one
frozen object is what lets a deployment (:mod:`control.live`) rebuild the
*identical* stack a checkpoint was trained against from the configuration alone,
and what lets the command line (:mod:`control.dsac.cli`) expose every knob without
a hand-maintained list.
"""

from __future__ import annotations

from dataclasses import dataclass

from control.dsac.risk import CVaRRisk, MeanRisk, RiskMeasure
from control.world import WorldConfig
from economic_models.ground_truth.registry import DEFAULT_MODEL


@dataclass(frozen=True)
class TrainConfig:
    """Everything that defines one training run.

    The world fields mirror :class:`~control.world.WorldConfig`; ``proxy``,
    ``encoder`` and ``latent`` pick the world model and the conditioning latent it
    forecasts from; the remainder are the agent's optimisation knobs.
    ``total_steps`` is the number of environment steps and
    ``total_steps * branch_k`` the number of proxy draws, which is what dominates
    the wall clock.

    The two encoders are configured **independently**: ``encoder``/``latent`` is
    the filter the *world model* conditions on, ``obs_encoder``/``obs_latent`` the
    filter the *policy* carries as its memory. Nothing requires them to match --
    the observation must be computable on the ground-truth side, where there is no
    proxy to borrow a latent from, so the two are separate instances holding
    separate beliefs even when set to the same family. Crossing them (a Kalman
    world model read by an LSTM-memory policy, say) is a supported ablation, not a
    misconfiguration.
    """

    # -- world --
    #: which ground-truth economy to train against, a key of
    #: :data:`~economic_models.ground_truth.registry.MODELS`.
    model: str = DEFAULT_MODEL
    dt: float = 0.25  #: length of one step in years
    history_steps: int = 500  #: steps of the single run the proxy is fit on
    horizon: int = 50  #: steps per episode
    #: steps per exogenous future *in the bank*, defaulting to ``horizon``. The
    #: two come apart only for a deployment (:mod:`control.live`), which runs one
    #: future for hundreds of periods where training cuts the same bank into
    #: episodes of fifty -- and which needs the futures generated long enough to
    #: allow it, from the same world and the same seeds.
    world_horizon: int | None = None
    n_train_futures: int = 2000  #: exogenous paths for training episodes
    n_eval_futures: int = 64  #: held-out exogenous paths
    excitation: str = "realistic"  #: excitation preset for the world
    #: futures reserved for a live deployment, drawn under
    #: :attr:`deploy_excitation`. Zero leaves the bank empty and
    #: :func:`~control.live.deploy.rehearse` falls back on the eval one.
    n_deploy_futures: int = 0
    #: the preset that third bank is drawn under. ``None`` is the ordinary
    #: setting -- one economy throughout -- and anything else deploys the offline
    #: stack into a world it was not fitted in, which is the only condition under
    #: which an online correction has a gap to close. See
    #: :attr:`~control.world.WorldConfig.deploy_excitation`.
    deploy_excitation: str | None = None
    #: how far each deployment future's economy is redrawn around its preset.
    #: Non-zero makes every rehearsal a *different* economy of comparable
    #: difficulty, rather than another draw of the one the stack was built in.
    #: See :attr:`~control.world.WorldConfig.deploy_jitter`.
    deploy_jitter: float = 0.0
    #: how many rows of the history training episodes start from
    #: (:attr:`~control.world.WorldConfig.n_starts`). One is the original
    #: behaviour: every episode branches off the history's end, which is one draw
    #: from the economy's state distribution and reliably an extreme of
    #: something. Above one the *training* bank is spread over that many rows and
    #: the agent is optimised from a distribution of initial conditions instead of
    #: a point; evaluation and deployment still start at the end, so the score is
    #: comparable either way.
    n_starts: int = 1
    #: rows at the head of the history no start is taken from, so the encoder
    #: belief an episode begins with has converged
    #: (:attr:`~control.world.WorldConfig.start_burn_in`).
    start_burn_in: int = 100
    world_seed: int = 0  #: seeds the history and its futures

    # -- world model --
    proxy: str = "varx"  #: which proxy family is the environment
    #: widen the world model's sampled law where it is extrapolating
    #: (:attr:`~economic_models.proxy.models.varx.VARXProxy.epistemic`). The
    #: residual covariance a stochastic rollout draws from is one constant matrix,
    #: so it is as tight off-support as on it and an agent optimising against it
    #: is told nothing about having left the data. This scales each draw by the
    #: ridge's own predictive standard deviation, which grows with the query's
    #: leverage -- the cheapest honest signal available, and the one a risk-averse
    #: objective (``risk="cvar"``) turns into pessimism about the unknown.
    epistemic: bool = False
    #: the encoder whose filtered latent the proxy forecasts from (one of
    #: :data:`~control.dsac.builders.ENCODERS`). ``"kalman"`` is the
    #: linear-Gaussian filter, ``"lstm"`` its nonlinear sibling; ``"none"``
    #: leaves the proxy conditioned on the exogenous block alone, which is what
    #: the random walk is.
    encoder: str = "kalman"
    latent: int = 10  #: latent width of the proxy's state encoder

    # -- what the policy remembers --
    #: the encoder whose filtered latent is appended to the observation (one of
    #: :data:`~control.dsac.builders.ENCODERS`). At ``"none"`` the agent is
    #: memoryless: it sees only the current stationary features, which are
    #: Markov in a world that is not, since the excitation's hidden structural
    #: parameters persist unobserved.
    #: Anything else gives it a recursive summary of the episode so far, fit once
    #: on the history and frozen.
    obs_encoder: str = "none"
    obs_latent: int = 10  #: latent width of the observation encoder

    # -- mandate --
    pi_target: float = 0.02  #: annual inflation target
    collapse_penalty: float = -25.0  #: reward per remaining step of a collapsed episode
    #: the instrument's speed: the maximum the policy's full deflection moves a
    #: lever, as a fraction of its box **per year** (annualized so one number
    #: means one thing at every :attr:`dt`). Part of the **environment** the
    #: agent is trained and evaluated in, and the same instrument the references
    #: steer and a live deployment operates -- see
    #: :attr:`~control.env.EnvConfig.delta_rate`.
    delta_rate: float = 0.4

    # -- agent --
    total_steps: int = 100_000  #: environment steps (== proxy steps)
    warmup: int = 2_000  #: uniform-random steps before learning starts
    batch: int = 256  #: minibatch size
    capacity: int = 200_000  #: replay capacity
    gamma: float = 0.99  #: discount per step
    tau: float = 0.005  #: Polyak rate of the target critics
    lr: float = 3e-4  #: Adam learning rate
    n_quantiles: int = 32  #: quantiles per critic
    hidden: int = 256  #: width of every network body
    n_layers: int = 2  #: hidden layers per network body
    updates_per_step: int = 1  #: gradient steps per environment step
    #: draws of each transition (see
    #: :meth:`~control.env.CentralBankEnv.branch_step`). At ``1`` the buffer holds
    #: one sample of the kernel per visited state and the critic recovers its
    #: spread only by smoothing across neighbouring states; above ``1`` it holds
    #: ``branch_k`` samples at the state itself, at ``branch_k`` times the proxy
    #: draws for the same ``total_steps`` of trajectory. Worth its cost only where
    #: that smoothing is thin, which is the tail ``risk="cvar"`` optimises over.
    branch_k: int = 1
    risk: str = "mean"  #: "mean" (risk-neutral) or "cvar"
    cvar_alpha: float = 0.1  #: tail fraction when ``risk="cvar"``

    # -- bookkeeping --
    #: end the run on the best-scoring checkpoint rather than the last one. The
    #: agent is training against a *surrogate*, and the longer it trains the more
    #: of the surrogate's error it is free to exploit -- so the held-out score
    #: rises, peaks, and then falls while the training score keeps improving. The
    #: last iterate is not the best iterate, and there is no reason to deploy it.
    #:
    #: The cost is that the held-out futures, which training never learned from,
    #: are now used to *select*, so the reported evaluation return is mildly
    #: optimistic. The ground-truth replay is unaffected: it runs on futures past
    #: the evaluation head (see :func:`~control.dsac.evaluation.rollout`'s ``first``), which no sweep and
    #: therefore no selection has ever touched.
    restore_best: bool = True
    eval_every: int = 5_000  #: environment steps between evaluations
    eval_episodes: int = 32  #: held-out futures per evaluation
    #: rolls per future during training evaluations. One is enough for a curve
    #: (the sweeps are paired, so the comparison is fair) and costs the same
    #: currency as training: a sweep is ``eval_episodes * horizon * repeats``
    #: proxy steps.
    eval_repeats: int = 1
    #: rolls per future for the final scoring, where separating "which future"
    #: from "how the proxy reacted" is worth paying for.
    final_repeats: int = 8
    log_every: int = 1_000  #: environment steps between progress lines
    seed: int = 0  #: seeds the agent, the replay sampling and the environments
    device: str | None = None  #: torch device (``None``/``"cpu"``/``"auto"``)
    #: intra-op torch threads, applied process-wide by
    #: :func:`~control.dsac.train.train` (``0`` leaves torch's default alone).
    #: One is the *fast* setting, not a throttle: the
    #: networks here are small enough that a batch of 256 through a width-256 MLP
    #: costs less than synchronising the threads that split it, so torch's default
    #: of one thread per core makes an update roughly twice as slow as running it
    #: on a single core. Staying single-threaded also leaves the other cores free,
    #: which is what makes a sweep of runs scale: independent processes at one
    #: thread each are far and away the cheapest speed-up available here.
    threads: int = 1

    def world_config(self) -> WorldConfig:
        """The :class:`~control.world.WorldConfig` this run's world is built from."""
        return WorldConfig(
            model=self.model,
            dt=self.dt,
            history_steps=self.history_steps,
            horizon=self.world_horizon or self.horizon,
            n_train_futures=self.n_train_futures,
            n_eval_futures=self.n_eval_futures,
            excitation=self.excitation,
            n_deploy_futures=self.n_deploy_futures,
            deploy_excitation=self.deploy_excitation,
            deploy_jitter=self.deploy_jitter,
            n_starts=self.n_starts,
            start_burn_in=self.start_burn_in,
            seed=self.world_seed,
        )

    def risk_measure(self) -> RiskMeasure:
        """The risk functional the actor maximises."""
        if self.risk == "mean":
            return MeanRisk()
        if self.risk == "cvar":
            return CVaRRisk(self.cvar_alpha)
        raise ValueError(f"risk must be 'mean' or 'cvar', got {self.risk!r}")
