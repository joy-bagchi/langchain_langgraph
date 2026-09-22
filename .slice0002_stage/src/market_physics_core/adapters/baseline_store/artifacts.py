"""Deterministic serialization for baseline history bundles."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from hashlib import sha256
from io import StringIO
import json
from typing import Any


ARTIFACT_NAMES = ("summary.json", "daily_skew.csv", "report.md")


def sha256_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# SPY VIX-scaled skew baseline", "",
        f"As of: {summary['as_of_session']}",
        f"Methodology: {summary['methodology_version']}",
        f"Status: {summary['status']}",
        f"Window: {summary['window_start']} through {summary['window_end']}", "",
    ]
    for group in summary["dte_groups"]:
        stats = group["statistics"]
        weekly = group["weekly_statistics"]
        latest = group["latest"]
        lines.extend([
            f"## {group['trading_session_dte']}-DTE", "",
            f"Eligibility: {group.get('eligibility_status', group['coverage'])}",
            f"Usable: {group['usable_n']} of {group['available_surface_dates']} available surface dates "
            f"({group['nominal_session_count']} nominal sessions)",
            f"Rolling mean / median: {stats['mean']} / {stats['median']}",
            f"Rolling sample SD: {stats['sample_standard_deviation']}",
            f"Rolling Q1 / Q3: {stats['q1']} / {stats['q3']}",
            f"Rolling range: {stats['minimum']} to {stats['maximum']}",
            f"Week mean / median: {weekly['mean']} / {weekly['median']}",
            f"Week usable / expected: {group['weekly_usable_n']} / {group['weekly_expected_sessions']}",
            f"Week complete: {group['week_complete']}",
            f"Exclusions: {json.dumps(group['exclusions'], sort_keys=True)}",
            (f"Latest: {latest['skew_iv_percentage_points']} on {latest['session']}; "
             f"difference from prior-days mean: {latest['difference_from_prior_mean']}"
             if latest else "Latest: unavailable"), "",
        ])
    lines.extend([
        "Successive rolling windows overlap; their mean changes are not independent observations.",
        "", "© Cloudpulse Innovations", "",
    ])
    return "\n".join(lines)


def serialize_bundle(
    summary: dict[str, Any], observations: list[dict[str, Any]], *, generated_at: str | None = None
) -> tuple[dict[str, Any], dict[str, bytes]]:
    value = json.loads(json.dumps(summary))
    buffer = StringIO(newline="")
    fieldnames = sorted({key for row in observations for key in row})
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(observations)
    csv_payload = buffer.getvalue().encode("utf-8")
    value["generated_at"] = generated_at or datetime.now(timezone.utc).isoformat()
    value["observation_artifact"] = {
        "path": "daily_skew.csv", "sha256": sha256_bytes(csv_payload)
    }
    summary_payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    report_payload = render_report(value).encode("utf-8")
    return value, {
        "summary.json": summary_payload,
        "daily_skew.csv": csv_payload,
        "report.md": report_payload,
    }


def bundle_relative_path(summary: dict[str, Any]) -> str:
    multiplier = f"sigma_{summary['sigma_multiplier']:g}".replace("-", "m").replace(".", "p")
    return "/".join(("baselines", summary["metric_id"], multiplier,
                     summary["methodology_version"], summary["as_of_session"], summary["run_id"]))


__all__ = ["ARTIFACT_NAMES", "bundle_relative_path", "render_report", "serialize_bundle", "sha256_bytes"]
