"""``python -m control``: one training run, configured from the command line.

Every :class:`~control.dsac.config.TrainConfig` field becomes a flag, read off the
dataclass rather than listed by hand, so a knob added to the configuration is on
the command line without anyone remembering to put it there.
"""

from __future__ import annotations

import argparse
import io
import sys
from collections.abc import Callable
from dataclasses import fields, replace
from typing import cast

from control.dsac.config import TrainConfig
from control.dsac.train import train


def _flag_kind(annotation: str) -> Callable[[str], object]:
    """The parser for one config field, from its declared type.

    Read off the **annotation** rather than the default's runtime type, which is
    what a field defaulting to ``None`` has none of: ``int | None`` would parse as
    ``str`` and hand the run ``"210"`` where it expected ``210``. ``"none"`` is
    accepted for optional fields so a value can be cleared from the command line.
    """
    base = annotation.replace("| None", "").replace("Optional", "").strip(" []")
    parse: Callable[[str], object] = {"int": int, "float": float}.get(base, str)
    if "None" not in annotation:
        return parse
    return lambda text: None if text.lower() in ("none", "") else parse(text)


def _parse_args() -> TrainConfig:
    """Build a :class:`~control.dsac.config.TrainConfig` from the command line.

    Every field becomes a flag. Booleans become a ``--flag/--no-flag`` pair,
    because ``type=bool`` in argparse is a trap -- it runs the constructor on the
    string, and every non-empty string is true, so ``--restore-best False`` would
    switch the option *on*.
    """
    defaults = TrainConfig()
    ap = argparse.ArgumentParser(
        description="Train a DSAC central bank inside a fitted proxy economy."
    )
    for f in fields(defaults):
        flag = "--" + f.name.replace("_", "-")
        current = getattr(defaults, f.name)
        help_text = f.metadata.get("help", "")
        if isinstance(current, bool):
            ap.add_argument(
                flag,
                default=current,
                help=help_text,
                action=argparse.BooleanOptionalAction,
            )
            continue
        ap.add_argument(
            flag, type=_flag_kind(str(f.type)), default=current, help=help_text
        )
    args = ap.parse_args()
    return replace(
        defaults, **{f.name: getattr(args, f.name) for f in fields(defaults)}
    )


def main() -> None:
    """Train an agent from the command line and report the final scores.

    A training run is hours long and reports as it goes, but Python block-buffers
    stdout whenever it is not a terminal -- so redirected to a log file (which is
    how a run this long is usually started) the progress lines sit in the buffer
    and the file stays empty until the process exits. Line buffering costs
    nothing at one line per few thousand steps and makes ``tail -f`` work.
    """
    cast(io.TextIOWrapper, sys.stdout).reconfigure(line_buffering=True)
    config = _parse_args()
    result = train(config)
    scores = result.final_ or {"agent": result.evals_[-1], **result.baselines_}
    print(
        f"final: agent {scores['agent']['return']:.1f} "
        f"(taylor {scores['taylor']['return']:.1f}, "
        f"calibration {scores['calibration']['return']:.1f}, "
        f"random {scores['random']['return']:.1f})"
    )
