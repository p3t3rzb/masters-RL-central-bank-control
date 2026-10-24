"""Construction: the world model, the encoders and the environments a run steps.

Every name a configuration can carry -- a proxy family, an encoder family -- is
resolved to an object here and nowhere else, so the training loop, a deployment
and a hyperparameter study that rebuild "the same stack" cannot drift apart in
how they read the same :class:`~control.dsac.config.TrainConfig`. The two
environment builders differ only in the driver: everything a policy touches is
shared, which is what makes a proxy-trained policy transferable to the ground
truth unchanged.
"""

from __future__ import annotations

import copy

from control.drivers import GroundTruthDriver, ModelDriver, ProxyDriver
from control.dsac.config import TrainConfig
from control.env import CentralBankEnv, EnvConfig
from control.observation import Observer
from control.rewards import RewardFunction
from control.world import TrainingWorld
from economic_models.encoders import (
    KalmanEncoder,
    LSTMEncoder,
    NullEncoder,
    StateEncoder,
)
from economic_models.ground_truth.registry import DEFAULT_MODEL, ground_truth
from economic_models.interface import ModelInterface
from economic_models.proxy import (
    BaseProxyModel,
    DRFProxy,
    KNNProxy,
    MDNProxy,
    RandomWalkProxy,
    VARXProxy,
)

#: The proxy families this trainer can use as a world model.
PROXIES = ("varx", "drf", "mdn", "knn", "random-walk")

#: The state encoders, by name. Two are learned filters of the same shape -- the
#: linear-Gaussian :class:`~economic_models.encoders.kalman.KalmanEncoder` and its
#: nonlinear sibling the :class:`~economic_models.encoders.lstm.LSTMEncoder`, both
#: taking a latent width and a seed and nothing else -- so either can play either
#: of the two encoder roles here. ``"none"`` is the zero-width
#: :class:`~economic_models.encoders.base.NullEncoder`: as the *proxy's* encoder
#: it leaves a forecaster conditioned on the exogenous block alone (which is what
#: the random walk uses), and as the *observation's* it is the memoryless agent.
ENCODERS = ("none", "kalman", "lstm")


# -- construction ----------------------------------------------------------


def build_encoder(
    name: str, *, latent: int = 10, seed: int = 0, role: str = "encoder"
) -> StateEncoder:
    """An unfitted state encoder of family ``name`` (one of :data:`ENCODERS`).

    The one place an encoder is named, for both roles: the latent a proxy
    forecasts from and the latent the observation carries. ``latent`` is its
    width, ``seed`` its initialisation, and ``role`` only names the field in the
    error message, since the two are configured separately and either can be
    wrong.

    Nothing beyond the width and the seed is exposed. The encoders' remaining
    knobs (the LSTM's depth, epochs and learning rate, say) are the encoder
    family's own business and are left at their defaults here, which is what keeps
    the two families interchangeable at this seam.
    """
    if name not in ENCODERS:
        raise ValueError(f"{role} must be one of {ENCODERS}, got {name!r}")
    if name == "kalman":
        return KalmanEncoder(latent_dim=latent, seed=seed)
    if name == "lstm":
        return LSTMEncoder(latent_dim=latent, seed=seed)
    return NullEncoder()


def build_proxy(
    name: str,
    *,
    interface: ModelInterface | None = None,
    encoder: str | StateEncoder = "kalman",
    latent: int = 10,
    seed: int = 0,
    epistemic: bool = False,
) -> BaseProxyModel:
    """An unfitted proxy of family ``name`` over ``interface``.

    ``interface`` is the ground-truth model the proxy stands in for, defaulting
    to GROWTH's so existing callers are unaffected.

    ``encoder`` picks the family of the conditioning latent it forecasts from and
    ``latent`` that latent's width -- a free choice, since a proxy reads its
    encoder through the :class:`~economic_models.encoders.base.StateEncoder`
    interface alone. An already-built encoder may be passed instead of a name, in
    which case it is **copied**: that is how a caller gets several proxies that
    agree on what their latent means (cross-fitting needs exactly this), without
    them sharing one belief.

    Either way each proxy gets its own instance, never a shared one: an encoder
    is fitted state, and two proxies fitted on the same history must not end up
    filtering into the same object.

    The random walk takes no encoder (it is the encoder-free baseline) and ignores
    both arguments.

    ``epistemic`` widens the sampled law by the fitted map's own parameter
    uncertainty where a query leaves the design
    (:attr:`~economic_models.proxy.models.varx.VARXProxy.epistemic`). Only the
    linear family has it in closed form, so it is ignored by the others rather
    than silently approximated.
    """
    interface = interface or ground_truth(DEFAULT_MODEL).interface
    if name not in PROXIES:
        raise ValueError(f"proxy must be one of {PROXIES}, got {name!r}")

    def fresh_encoder() -> StateEncoder:
        """This proxy's own encoder: a copy of a built one, or a new one by name."""
        if isinstance(encoder, StateEncoder):
            return copy.deepcopy(encoder)
        return build_encoder(encoder, latent=latent, seed=seed, role="encoder")

    if name == "varx":
        return VARXProxy(interface, encoder=fresh_encoder(), epistemic=epistemic)
    if name == "drf":
        return DRFProxy(interface, encoder=fresh_encoder(), seed=seed)
    if name == "mdn":
        return MDNProxy(interface, encoder=fresh_encoder(), seed=seed)
    if name == "knn":
        return KNNProxy(interface, encoder=fresh_encoder())
    return RandomWalkProxy(interface)


def build_obs_encoder(name: str, *, latent: int = 10, seed: int = 0) -> StateEncoder:
    """An unfitted state encoder for the observation's latent block.

    Deliberately a fresh instance rather than the proxy's, even where both are the
    same family: the observation has to be computable on the ground-truth side
    too, where there is no proxy to borrow a component from. Both end up fit on
    the same history, so they agree.
    """
    return build_encoder(name, latent=latent, seed=seed, role="obs_encoder")


def _env_over(
    driver: ModelDriver,
    world: TrainingWorld,
    observer: Observer,
    reward: RewardFunction,
    config: TrainConfig,
    *,
    split: str,
    seed: int,
    horizon: int | None = None,
) -> CentralBankEnv:
    """An environment over ``driver``, drawing from the train or eval futures."""
    spec = world.config.spec
    episodes = world.train_futures if split == "train" else world.eval_futures
    return CentralBankEnv(
        driver,
        episodes,
        reward,
        observer,
        spec.interface,
        EnvConfig(
            collapse_penalty=config.collapse_penalty,
            horizon=horizon or config.horizon,
            delta_rate=config.delta_rate,
            action_bounds=spec.action_bounds,
            er_bounds=spec.er_bounds,
            pi_bounds=spec.pi_bounds,
        ),
        seed=seed,
    )


def build_env(
    proxy: BaseProxyModel,
    world: TrainingWorld,
    observer: Observer,
    reward: RewardFunction,
    config: TrainConfig,
    *,
    split: str,
    seed: int,
) -> CentralBankEnv:
    """An environment over ``proxy`` drawing from the train or eval futures."""
    return _env_over(
        ProxyDriver(proxy, world),
        world,
        observer,
        reward,
        config,
        split=split,
        seed=seed,
    )


def build_truth_env(
    world: TrainingWorld,
    observer: Observer,
    reward: RewardFunction,
    config: TrainConfig,
    *,
    split: str = "eval",
    seed: int = 0,
    horizon: int | None = None,
) -> CentralBankEnv:
    """The same environment over the **structural** model instead of a proxy.

    Everything a policy touches is unchanged -- the same observer, the same
    mandate, the same action box, the same futures -- so a policy trained in the
    proxy can be dropped straight in and what differs is only who answers "what
    happens next". That is the whole point: the gap between a policy's return here
    and in the proxy it was trained in *is* the surrogate's error, expressed in the
    units the mandate is written in.

    Expensive by comparison -- every step solves the full system rather than
    sampling a fitted conditional -- and not branchable, so it is for evaluation
    and never for training.

    ``horizon`` overrides how many steps of a drawn future an episode runs for,
    which a deployment needs: it takes one future for hundreds of periods where
    training cut the bank into episodes of fifty.
    """
    return _env_over(
        GroundTruthDriver(world),
        world,
        observer,
        reward,
        config,
        split=split,
        seed=seed,
        horizon=horizon,
    )
