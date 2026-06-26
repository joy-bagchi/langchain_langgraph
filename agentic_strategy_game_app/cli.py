"""CLI entrypoint for the agentic strategy game app."""

from __future__ import annotations

import argparse
import json

from agentic_strategy_game_app.scenarios import (
    build_b2b_saas_ai_disruption_scenario,
    list_scenarios,
)


def _render_scenario(name: str) -> dict:
    if name != "b2b_saas_ai_disruption":
        raise ValueError(f"Unknown scenario '{name}'. Available: {', '.join(list_scenarios())}")
    config, world = build_b2b_saas_ai_disruption_scenario()
    return {"simulation_config": config.to_dict(), "initial_world_state": world.to_dict()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect strategy game scenarios.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-scenarios", help="List available strategy-game scenarios.")

    describe_parser = subparsers.add_parser("describe-scenario", help="Print a scenario contract as JSON.")
    describe_parser.add_argument("--name", default="b2b_saas_ai_disruption")

    args = parser.parse_args()

    if args.command == "list-scenarios":
        print(json.dumps({"scenarios": list_scenarios()}, indent=2))
        return

    print(json.dumps(_render_scenario(args.name), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
