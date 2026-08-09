"""Command-line entry points for publishing and rendering IV cubes."""

from __future__ import annotations

import argparse

from .cube import create_iv_cube_figure
from .reader import DEFAULT_BUCKET, DEFAULT_PREFIX

from .reader import load_option_chain_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish and visualize IBKR implied-volatility cubes.")
    sub = parser.add_subparsers(dest="command", required=True)
    render = sub.add_parser("render", help="Load the published GCS catalog and write an interactive HTML cube.")
    render.add_argument("--bucket", default=DEFAULT_BUCKET); render.add_argument("--prefix", default=DEFAULT_PREFIX)
    render.add_argument("--project"); render.add_argument("--right", choices=("C", "P")); render.add_argument("--start-date"); render.add_argument("--end-date"); render.add_argument("--output", default="iv_cube.html")
    args = parser.parse_args()
    create_iv_cube_figure(load_option_chain_history(bucket=args.bucket, prefix=args.prefix, project=args.project, start_date=args.start_date, end_date=args.end_date), right=args.right).write_html(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
