"""Internal Torch helpers shared by the neural estimators (LSTM, MDN).

Private to the package. Kept separate so the pure-numpy estimators (Kalman,
DRF, ...) never import Torch transitively.
"""

from __future__ import annotations

import torch


def resolve_device(device: str | torch.device | None) -> torch.device:
    """Resolve a device spec to a concrete :class:`torch.device`.

    ``None`` or ``"cpu"`` -> CPU (the reproducible default: results match the
    reference CPU run bit-for-bit). ``"auto"`` picks the best available
    accelerator -- CUDA, then Apple MPS, then CPU -- so callers can opt into GPU
    acceleration on either platform with one string. Any other value (``"cuda"``,
    ``"mps"``, ``"cuda:1"``, ...) is passed through to :class:`torch.device`.

    Note: accelerated backends reduce in a different order than CPU BLAS, so a
    run on ``"auto"``/``"mps"``/``"cuda"`` matches the CPU reference only to
    float32 tolerance, not bit-for-bit. CPU stays the default for that reason.
    """
    if device is None:
        return torch.device("cpu")
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)
