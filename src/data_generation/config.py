"""The :class:`DatasetConfig`: everything that fixes *which* dataset is generated.

A config is a small, picklable value object (so it can be shipped to worker
processes unchanged) describing the ensemble to produce -- how many main
histories, how many branching continuations each, how long they run, the
excitation regime, the reproducibility seed and the train/test split.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

#: The excitation presets a dataset may be generated under, mapped to the
#: :class:`~economic_models.ground_truth.GrowthExcitationConfig` classmethod that
#: builds them. ``"default"`` is the calm, full-length-guaranteed calibration;
#: ``"realistic"`` layers volatility clustering and (recoverable) crises on top.
EXCITATIONS = ("default", "realistic")


@dataclass(frozen=True)
class DatasetConfig:
    """A fully-specified GROWTH-model dataset to generate.

    ``n_runs`` independent *main* histories are simulated, each excited for
    ``main_length`` recorded steps; from the end of every main run,
    ``n_continuations`` alternative futures branch off, each ``continuation_length``
    steps long. The main runs are split into training and testing groups at the
    group level (a main run and all its continuations stay on the same side of
    the split), so no continuation leaks across it.
    """

    n_runs: int = 1000  #: number of independent main (historic) runs
    n_continuations: int = 100  #: continuations branching from each main run's end
    main_length: int = 250  #: recorded steps in each main run
    continuation_length: int = 50  #: recorded steps in each continuation
    dt: float = 0.25  #: length of one step in years (1.0 annual, 1/12 monthly)
    excitation: str = "default"  #: excitation preset, one of :data:`EXCITATIONS`
    burn_in: int = 15  #: years run unrecorded to settle each main run onto its path
    on_collapse: str = (
        "resample"  #: 'raise', 'truncate' or 'resample' on unrecoverable solver failure
    )
    train_fraction: float = (
        0.8  #: fraction of main runs (and their continuations) used for training
    )
    base_seed: int = 0  #: root seed; every run/continuation seed derives from it
    output_dir: str = "data"  #: directory the dataset is written under (next to src/)
    workers: int | None = None  #: worker processes (defaults to the CPU count)

    def __post_init__(self) -> None:
        """Validate the configuration eagerly, before any expensive generation."""
        if self.excitation not in EXCITATIONS:
            raise ValueError(
                f"excitation must be one of {EXCITATIONS}, got {self.excitation!r}"
            )
        if self.on_collapse not in ("raise", "truncate", "resample"):
            raise ValueError("on_collapse must be 'raise', 'truncate' or 'resample'")
        if not 0.0 <= self.train_fraction <= 1.0:
            raise ValueError("train_fraction must be in [0, 1]")
        for name in ("n_runs", "n_continuations", "main_length", "continuation_length"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")

    @property
    def n_train(self) -> int:
        """Number of main runs assigned to the training split."""
        return round(self.n_runs * self.train_fraction)

    @property
    def n_test(self) -> int:
        """Number of main runs assigned to the testing split."""
        return self.n_runs - self.n_train

    def resolved_workers(self) -> int:
        """The worker-process count to use (the CPU count when unset)."""
        return self.workers if self.workers is not None else (os.cpu_count() or 1)
