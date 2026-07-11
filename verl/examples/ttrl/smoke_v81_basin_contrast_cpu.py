#!/usr/bin/env python3
"""CPU smoke checks for v81 basin-contrast target calibration."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path("/opt/tiger/TTRL/verl")
sys.path.insert(0, str(ROOT))

from verl.trainer.ppo.ttrl_utils import _apply_basin_contrast_calibration  # noqa: E402


def assert_close(name: str, value: float, expected: float, tol: float = 1e-6) -> None:
    if abs(value - expected) > tol:
        raise AssertionError(f"{name}: got={value} expected={expected}")


def assert_prob_vector(name: str, probs: torch.Tensor) -> None:
    if torch.any(probs < -1e-8):
        raise AssertionError(f"{name}: negative probability {probs}")
    assert_close(f"{name}.sum", float(probs.sum().item()), 1.0)


def main() -> None:
    target = torch.tensor([0.90, 0.06, 0.04], dtype=torch.float32)
    weak_base = torch.tensor([0.34, 0.33, 0.33], dtype=torch.float32)
    weak_low = torch.tensor([0.34, 0.33, 0.33], dtype=torch.float32)
    weak_process = torch.tensor([0.34, 0.33, 0.33], dtype=torch.float32)
    weak_support = torch.tensor([0.34, 0.33, 0.33], dtype=torch.float32)
    calibrated, alpha, target_margin, support_margin, base_margin, low_margin, proc_margin = (
        _apply_basin_contrast_calibration(
            target,
            weak_base,
            weak_low,
            weak_process,
            weak_support,
            strength=0.8,
            margin=0.12,
        )
    )
    assert_prob_vector("weak.calibrated", calibrated)
    if alpha <= 0.0:
        raise AssertionError(f"weak evidence should activate calibration, alpha={alpha}")
    if float(calibrated[0].item()) >= float(target[0].item()):
        raise AssertionError("weak evidence should reduce top mass")
    if int(torch.argmax(calibrated).item()) != 0:
        raise AssertionError("basin contrast should calibrate softly, not replace the top basin")
    if not (target_margin > support_margin and target_margin > base_margin):
        raise AssertionError("weak case should have target margin exceeding support/base margin")
    if min(base_margin, low_margin, proc_margin) < -1e-8:
        raise AssertionError("weak margins should be near-zero positive")

    strong_support = torch.tensor([0.86, 0.10, 0.04], dtype=torch.float32)
    unchanged, alpha, *_ = _apply_basin_contrast_calibration(
        target,
        strong_support,
        strong_support,
        strong_support,
        strong_support,
        strength=0.8,
        margin=0.12,
    )
    assert_prob_vector("strong.unchanged", unchanged)
    assert_close("strong.alpha", alpha, 0.0)
    if not torch.allclose(unchanged, target, atol=1e-6):
        raise AssertionError(f"strong evidence should leave target unchanged: {unchanged}")

    disabled, alpha, *_ = _apply_basin_contrast_calibration(
        target,
        weak_base,
        weak_low,
        weak_process,
        weak_support,
        strength=0.0,
        margin=0.12,
    )
    assert_prob_vector("disabled", disabled)
    assert_close("disabled.alpha", alpha, 0.0)
    if not torch.allclose(disabled, target, atol=1e-6):
        raise AssertionError("strength=0 must be exact no-op")

    print("V81_BASIN_CONTRAST_CPU_SMOKE_OK")


if __name__ == "__main__":
    main()
