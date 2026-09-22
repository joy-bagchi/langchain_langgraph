"""Immutable local baseline-history writer."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# SPY VIX-scaled skew baseline",
        "",
        f"As of: {summary['as_of_session']}",
        f"Methodology: {summary['methodology_version']}",
        f"Status: {summary['status']}",
        f"Window: {summary['window_start']} through {summary['window_end']}",
        "",
    ]
    for group in summary["dte_groups"]:
        stats = group["statistics"]
        latest = group["latest"]
        lines.extend(
            [
                f"## {group['trading_session_dte']}-DTE",
                "",
                f"Usable: {group['usable_n']} of {group['available_surface_dates']} available surface dates "
                f"({group['nominal_session_count']} nominal sessions)",
                f"Mean: {stats['mean']}",
                f"Median: {stats['median']}",
                f"Sample standard deviation: {stats['sample_standard_deviation']}",
                f"Q1 / Q3: {stats['q1']} / {stats['q3']}",
                f"Range: {stats['minimum']} to {stats['maximum']}",
                f"Exclusions: {json.dumps(group['exclusions'], sort_keys=True)}",
                (
                    "Latest: "
                    f"{latest['skew_iv_percentage_points']} on {latest['session']}; "
                    f"difference from prior-days mean: {latest['difference_from_prior_mean']}"
                    if latest
                    else "Latest: unavailable"
                ),
                "",
            ]
        )
    lines.extend(["© Cloudpulse Innovations", ""])
    return "\n".join(lines)


def write_local_history(
    root: Path, summary: dict[str, Any], observations: list[dict[str, Any]]
) -> tuple[Path, dict[str, str]]:
    multiplier_id = (
        f"sigma_{summary['sigma_multiplier']:g}".replace("-", "m").replace(".", "p")
    )
    run_dir = (
        root
        / "baselines"
        / summary["metric_id"]
        / multiplier_id
        / summary["methodology_version"]
        / summary["as_of_session"]
        / summary["run_id"]
    )
    if run_dir.exists():
        required = [run_dir / name for name in ("summary.json", "daily_skew.csv", "report.md")]
        if not all(path.is_file() for path in required):
            raise ValueError(f"incomplete existing immutable run: {run_dir}")
        return run_dir, {path.name: _sha(path) for path in required}
    run_dir.mkdir(parents=True, exist_ok=False)
    csv_path = run_dir / "daily_skew.csv"
    fieldnames = sorted({key for row in observations for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(observations)
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["observation_artifact"] = {
        "path": "daily_skew.csv",
        "sha256": _sha(csv_path),
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report_path = run_dir / "report.md"
    report_path.write_text(_report(summary), encoding="utf-8")
    paths = (summary_path, csv_path, report_path)
    checksums = {path.name: _sha(path) for path in paths}
    for path in paths:
        if _sha(path) != checksums[path.name]:
            raise ValueError(f"read-back checksum verification failed: {path}")
    return run_dir, checksums


__all__ = ["write_local_history"]
