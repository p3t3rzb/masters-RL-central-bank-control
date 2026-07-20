"""The GROWTH-specific excitation spec: the government-spending stabilizer.

The model-agnostic specs (:class:`AR1Spec`, :class:`RandomWalkSpec`,
:class:`CrisisSpec`, ...) live in :mod:`economic_models.ground_truth.excitation.specs`;
this module adds only the spec that encodes GROWTH's own fiscal stabilizer.
"""

from __future__ import annotations

from dataclasses import dataclass

from economic_models.ground_truth.excitation.specs import AR1Spec


@dataclass(frozen=True)
class GovSpendingSpec:
    """Excitation of government spending growth ``GRg``.

    ``GRg`` is not excited independently but as a bounded *gap to productivity
    growth*: trend demand must track trend supply, or a persistent gap (e.g.
    ``GRg`` stuck near 0 while ``GRpr`` is 3%) puts the economy in a permanent
    demand depression no monetary policy can offset. On top of that AR(1) gap, a
    symmetric countercyclical stabilizer moves spending against the employment
    gap. This is the model's only fast, reliable stabilizer: it guards against
    the self-reinforcing wage-deflation spiral below ``ER ~ 0.9`` and damps
    overheating booms above full employment.
    """

    gap_sigma: float  #: innovation std of the AR(1) gap to productivity growth
    gap_clip: float  #: maximum absolute gap to productivity growth
    stabilizer: float  #: countercyclical gain on the employment gap ``1 - ER``
    bounds: tuple[float, float]  #: hard clip on the resulting growth rate

    def gap_spec(self) -> AR1Spec:
        """The gap to productivity growth as an :class:`AR1Spec` around 0."""
        return AR1Spec(sigma=self.gap_sigma, lower=-self.gap_clip, upper=self.gap_clip)
