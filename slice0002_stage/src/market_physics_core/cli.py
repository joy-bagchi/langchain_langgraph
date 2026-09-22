"""Command-line entry point for market-physics-core."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import subprocess
import tomllib

from market_physics_core import __version__
from market_physics_core.adapters.baseline_store.local import write_local_history
from market_physics_core.adapters.market_data.gcs import GCSMarketDataReader
from market_physics_core.applications.skew_baseline.service import calculate_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="market-physics-core",
        description="Deterministic analysis engine for Market Physics.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    baseline = subparsers.add_parser(
        "skew-baseline",
        help="Calculate and retain the SPY VIX-scaled skew baseline.",
    )
    baseline.add_argument(
        "--config",
        type=Path,
        default=Path("configs/skew_baseline.example.toml"),
    )
    baseline.add_argument("--output-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "skew-baseline":
        with args.config.open("rb") as stream:
            config = tomllib.load(stream)
        inputs_config = config["inputs"]
        market_inputs = GCSMarketDataReader(
            project=inputs_config["gcp_project"],
            bucket=inputs_config["gcs_bucket"],
            option_catalog_object=inputs_config["option_catalog_object"],
            volatility_history_manifest_object=inputs_config[
                "volatility_history_manifest_object"
            ],
        ).load()
        code_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        summary, observations = calculate_baseline(
            market_inputs,
            code_commit=code_commit,
            sigma_multiplier=float(config["sigma_multiplier"]),
            methodology_version=str(config["methodology_version"]),
            rolling_window_sessions=int(config["rolling_window_sessions"]),
            calendar_name=str(config["exchange_calendar"]),
        )
        output_root = args.output_root or Path(config["outputs"]["local_history_path"])
        run_dir, checksums = write_local_history(
            output_root, summary, observations
        )
        print(
            json.dumps(
                {
                    "run_directory": str(run_dir.resolve()),
                    "checksums": checksums,
                    "summary": summary,
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
