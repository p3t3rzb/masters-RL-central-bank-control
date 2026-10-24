"""On-disk format for generated datasets.

A group on disk holds three kinds of artifact, each a compressed ``.npz``:

* the **main run** -- a fully simulated :class:`~economic_models.run.Run`
  (states/params/actions/hidden + diagnostics), the historic record;
* the **branch state** -- the full internal ground-truth model state the main run
  ends in, the shared starting point every continuation is rolled out from;
* the **continuation scenarios** -- :class:`~economic_models.run.Scenario`
  objects holding only the exogenous forcing (params + hidden), the world an
  in-control agent acts within (it supplies the actions; the model then produces
  the states).

Each artifact round-trips losslessly through its ``save_*`` / ``load_*`` pair. The
column order of the arrays is fixed by the model interface and recorded once in
the top-level manifest written by :func:`write_manifest`.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from economic_models.run import Run, Scenario

#: Sentinel stored for an absent ``seed`` (``None``) in the integer metadata.
_NO_SEED = -1

#: The arrays a :class:`Run` / :class:`Scenario` may or may not carry, stored only
#: when present (in this order) and read back as ``None`` when absent.
_OPTIONAL_ARRAYS = ("hidden", "volatility", "crisis_intensity")


def _pack_meta(meta: dict[str, Any]) -> np.ndarray:
    """A JSON metadata block as a 0-d array (``np.savez`` can't store None/scalars)."""
    return np.asarray(json.dumps(meta))


def _prepare(path: str | Path) -> Path:
    """``path`` as a :class:`Path`, with its parent directory created."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _optional_arrays(record: Run | Scenario) -> dict[str, np.ndarray]:
    """The :data:`_OPTIONAL_ARRAYS` ``record`` actually carries, keyed by name."""
    present = {name: getattr(record, name) for name in _OPTIONAL_ARRAYS}
    return {name: array for name, array in present.items() if array is not None}


def _load_optional(data: Any, name: str) -> np.ndarray | None:
    """Array ``name`` from an open ``.npz``, or ``None`` if it was not stored."""
    return data[name] if name in data else None


def _encode_seed(seed: int | None) -> int:
    """A seed as stored in the metadata (:data:`_NO_SEED` for ``None``)."""
    return _NO_SEED if seed is None else int(seed)


def _decode_seed(stored: int) -> int | None:
    """The inverse of :func:`_encode_seed`."""
    return None if stored == _NO_SEED else int(stored)


def _encode_climate(climate: float | None) -> float:
    """A climate as stored in the metadata (``nan`` for ``None``)."""
    return np.nan if climate is None else float(climate)


def _decode_climate(stored: float) -> float | None:
    """The inverse of :func:`_encode_climate`."""
    return None if np.isnan(stored) else float(stored)


def save_run(path: str | Path, run: Run) -> None:
    """Write a fully-simulated :class:`Run` to ``path`` as a compressed ``.npz``.

    The states/params/actions/hidden arrays are stored under their own keys; the
    optional per-step diagnostics (volatility, crisis intensity) are stored only
    when present, and the scalar metadata (dt, seed, dampened steps, climate,
    collapse flag) travels in a small side block so the run reconstructs exactly.
    """
    path = _prepare(path)
    arrays: dict[str, np.ndarray] = {
        "states": run.states,
        "params": run.params,
        "actions": run.actions,
        **_optional_arrays(run),
    }
    arrays["_meta"] = _pack_meta(
        {
            "dt": float(run.dt),
            "seed": _encode_seed(run.seed),
            "dampened_steps": int(run.dampened_steps),
            "climate": _encode_climate(run.climate),
            "collapsed": bool(run.collapsed),
        }
    )
    np.savez_compressed(path, allow_pickle=False, **arrays)


def load_run(path: str | Path) -> Run:
    """Load a :class:`Run` previously written by :func:`save_run`."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["_meta"]))
        return Run(
            states=data["states"],
            params=data["params"],
            actions=data["actions"],
            hidden=_load_optional(data, "hidden"),
            dt=float(meta["dt"]),
            seed=_decode_seed(meta["seed"]),
            dampened_steps=int(meta["dampened_steps"]),
            volatility=_load_optional(data, "volatility"),
            crisis_intensity=_load_optional(data, "crisis_intensity"),
            climate=_decode_climate(meta["climate"]),
            collapsed=bool(meta["collapsed"]),
        )


def save_scenario(path: str | Path, scenario: Scenario) -> None:
    """Write a :class:`Scenario` (exogenous forcing only) to ``path`` as ``.npz``.

    Stores the params (and hidden) arrays and the diagnostics; no states or
    actions, since a scenario fixes neither -- the agent supplies the actions and
    the model then produces the states.
    """
    path = _prepare(path)
    arrays: dict[str, np.ndarray] = {
        "params": scenario.params,
        **_optional_arrays(scenario),
    }
    arrays["_meta"] = _pack_meta(
        {
            "dt": float(scenario.dt),
            "seed": _encode_seed(scenario.seed),
            "climate": _encode_climate(scenario.climate),
        }
    )
    np.savez_compressed(path, allow_pickle=False, **arrays)


def load_scenario(path: str | Path) -> Scenario:
    """Load a :class:`Scenario` previously written by :func:`save_scenario`."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["_meta"]))
        return Scenario(
            params=data["params"],
            hidden=_load_optional(data, "hidden"),
            dt=float(meta["dt"]),
            seed=_decode_seed(meta["seed"]),
            volatility=_load_optional(data, "volatility"),
            crisis_intensity=_load_optional(data, "crisis_intensity"),
            climate=_decode_climate(meta["climate"]),
        )


def save_branch_state(path: str | Path, state: dict[str, float]) -> None:
    """Write the full internal model ``state`` (name->value) to ``path`` as ``.npz``.

    The branch state pins the ground-truth model at the main run's end so a
    continuation scenario can be rolled out from exactly there; it holds every
    model variable (visible and hidden), not just the visible interface.
    """
    path = _prepare(path)
    names = np.asarray(list(state.keys()))
    values = np.asarray(list(state.values()), dtype=float)
    np.savez_compressed(path, names=names, values=values)


def load_branch_state(path: str | Path) -> dict[str, float]:
    """Load a branch state written by :func:`save_branch_state`, as a name->value dict."""
    with np.load(path, allow_pickle=False) as data:
        return {str(n): float(v) for n, v in zip(data["names"], data["values"])}


def group_dir(root: str | Path, split: str, index: int) -> Path:
    """The directory holding one group: its main run, branch state and scenarios.

    ``root/<split>/group_<index>``; ``split`` is ``"train"`` or ``"test"``.
    """
    return Path(root) / split / f"group_{index:04d}"


def main_path(root: str | Path, split: str, index: int) -> Path:
    """Path of a group's main run file."""
    return group_dir(root, split, index) / "main.npz"


def branch_path(root: str | Path, split: str, index: int) -> Path:
    """Path of a group's branch-state file (the shared continuation start)."""
    return group_dir(root, split, index) / "branch.npz"


def scenario_path(root: str | Path, split: str, index: int, j: int) -> Path:
    """Path of a group's ``j``-th continuation scenario file."""
    return group_dir(root, split, index) / f"scenario_{j:03d}.npz"


def write_manifest(root: str | Path, manifest: dict[str, Any]) -> None:
    """Write the dataset ``manifest`` to ``root/manifest.json``."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)


def read_manifest(root: str | Path) -> dict[str, Any]:
    """Read the dataset manifest written by :func:`write_manifest`."""
    with open(Path(root) / "manifest.json") as fh:
        return json.load(fh)


def config_to_dict(config: Any) -> dict[str, Any]:
    """A JSON-serialisable dict of a :class:`~data_generation.config.DatasetConfig`."""
    return asdict(config)
