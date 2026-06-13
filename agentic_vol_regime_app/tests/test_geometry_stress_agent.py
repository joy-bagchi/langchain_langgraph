from __future__ import annotations

import numpy as np
import pandas as pd

from src.agents.geometry_stress_agent import compute_geometry_stress


def _geometry_frame(rows: int = 300) -> pd.DataFrame:
    idx = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "avg_pairwise_corr_21d": 0.2 + (idx * 0.001),
            "first_eigenvalue_share_21d": 0.3 + (idx * 0.0008),
            "effective_rank_21d": 5.0 - (idx * 0.005),
            "log_det_corr_21d": -1.0 - (idx * 0.01),
        }
    )


def test_geometry_stress_is_bounded_zero_to_one() -> None:
    output = compute_geometry_stress(_geometry_frame(300))
    assert 0.0 <= output.geometry_stress_score <= 1.0
    assert set(output.geometry_components.keys()) == {
        "avg_corr_stress",
        "eigen_stress",
        "effective_rank_stress",
        "log_det_stress",
    }


def test_geometry_stress_is_high_for_high_corr_eigen_low_rank_and_low_log_det() -> None:
    frame = _geometry_frame(300)
    output = compute_geometry_stress(frame)
    assert output.geometry_stress_score > 0.7
    assert output.geometry_confirmation_level in {"geometry_confirming", "geometry_strong_confirmation"}


def test_geometry_stress_returns_neutral_with_warning_on_short_history() -> None:
    output = compute_geometry_stress(_geometry_frame(20))
    assert output.geometry_stress_score == 0.5
    assert output.warnings
