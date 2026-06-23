from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest.hmm_replay.replay_config import ReplayConfig

STRICT_REPLAY_PRIMARY_YEARS = 10.0
STRICT_REPLAY_WARMUP_YEARS = 3.0
STRICT_REPLAY_MIN_PRIMARY_COVERAGE_RATIO = 0.75
STRICT_REPLAY_MIN_REQUIRED_TRADING_DAYS = int(
    252.0 * (STRICT_REPLAY_WARMUP_YEARS + (STRICT_REPLAY_PRIMARY_YEARS * STRICT_REPLAY_MIN_PRIMARY_COVERAGE_RATIO))
)

REQUIRED_HMMV4_COLUMNS = [
    "date",
    "spy_close",
    "vix",
    "vvix",
    "vix_vix3m_ratio",
    "realized_vol_5d",
    "realized_vol_21d",
    "drawdown_21d",
    "trend_persistence_21d",
    "avg_pairwise_corr_21d",
    "first_eigenvalue_share_21d",
    "effective_rank_21d",
    "log_det_corr_21d",
    "regime_target",
]


@dataclass(slots=True)
class ReplayPreflightResult:
    ok: bool
    earliest_date: str
    latest_date: str
    requested_start_date: str
    requested_end_date: str
    actual_start_date: str
    actual_end_date: str
    missing_columns: list[str]
    messages: list[str]
    fallback_rate_limit: float


def run_replay_preflight(
    *,
    frame: pd.DataFrame,
    config: ReplayConfig,
    requested_start_date: str,
    requested_end_date: str,
) -> ReplayPreflightResult:
    earliest = str(frame["date"].min()) if not frame.empty else ""
    latest = str(frame["date"].max()) if not frame.empty else ""
    missing_columns = [column for column in REQUIRED_HMMV4_COLUMNS if column not in frame.columns]
    messages: list[str] = []
    ok = True

    requested_start = pd.to_datetime(requested_start_date).date()
    requested_end = pd.to_datetime(requested_end_date).date()
    actual_start = requested_start
    actual_end = min(requested_end, pd.to_datetime(latest).date()) if latest else requested_end

    if missing_columns:
        ok = False
        messages.append("Feature store is missing required HMMv4 columns: " + ", ".join(missing_columns))

    if config.require_10y_replay:
        required_coverage_start = pd.to_datetime("2013-01-01").date()
        if not earliest:
            ok = False
            messages.append("Feature store is empty; cannot run required 10-year HMMv4 replay.")
        else:
            earliest_date = pd.to_datetime(earliest).date()
            row_count = int(len(frame))
            if earliest_date > required_coverage_start and row_count < STRICT_REPLAY_MIN_REQUIRED_TRADING_DAYS:
                ok = False
                messages.append(
                    "10-year HMMv4 replay requires sufficient history depth. "
                    f"earliest_available_date={earliest_date.isoformat()}, "
                    f"required_coverage_start={required_coverage_start.isoformat()}, "
                    f"row_count={row_count}, minimum_required_rows={STRICT_REPLAY_MIN_REQUIRED_TRADING_DAYS}."
                )

    return ReplayPreflightResult(
        ok=ok,
        earliest_date=earliest,
        latest_date=latest,
        requested_start_date=requested_start.isoformat(),
        requested_end_date=requested_end.isoformat(),
        actual_start_date=actual_start.isoformat(),
        actual_end_date=actual_end.isoformat(),
        missing_columns=missing_columns,
        messages=messages,
        fallback_rate_limit=0.10 if config.require_10y_replay else 1.0,
    )


def write_preflight_failure_artifacts(
    *,
    output_dir: Path,
    preflight: ReplayPreflightResult,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "preflight_failure_report.md"
    missing_csv_path = output_dir / "preflight_missing_data.csv"
    report_lines = [
        "# Replay Preflight Failure",
        "",
        f"- requested_start_date: `{preflight.requested_start_date}`",
        f"- requested_end_date: `{preflight.requested_end_date}`",
        f"- actual_start_date: `{preflight.actual_start_date}`",
        f"- actual_end_date: `{preflight.actual_end_date}`",
        f"- earliest_feature_store_date: `{preflight.earliest_date}`",
        f"- latest_feature_store_date: `{preflight.latest_date}`",
        "",
        "## Failures",
    ]
    report_lines.extend(f"- {message}" for message in preflight.messages)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "earliest_date": preflight.earliest_date,
                "latest_date": preflight.latest_date,
                "requested_start_date": preflight.requested_start_date,
                "requested_end_date": preflight.requested_end_date,
                "actual_start_date": preflight.actual_start_date,
                "actual_end_date": preflight.actual_end_date,
                "missing_columns": " | ".join(preflight.missing_columns),
                "messages": " | ".join(preflight.messages),
            }
        ]
    ).to_csv(missing_csv_path, index=False)
    return report_path, missing_csv_path
