"""CLI entrypoint for the volatility regime app."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentic_vol_regime_app.app_runtime import default_agent_path, resume_daily_regime_run, run_daily_regime_agent
from agentic_vol_regime_app.data.ibkr_client import (
    IBKRConnectionConfig,
    IBKRDataPipe,
    IBKROptionChainRequest,
)


def _load_input(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _render_output(result: dict[str, Any], output_mode: str) -> str:
    if output_mode == "internal":
        return json.dumps(result, indent=2, sort_keys=True)
    daily_report = dict(result.get("named_outputs", {})).get("daily_report", {})
    if output_mode == "report" and isinstance(daily_report, dict):
        return str(daily_report.get("markdown", ""))
    return json.dumps(
        {
            "run_id": result.get("run_id"),
            "status": result.get("status"),
            "current_step": result.get("current_step"),
            "alert": dict(result.get("named_outputs", {})).get("alert_record", {}),
            "policy": dict(result.get("named_outputs", {})).get("policy_recommendation", {}),
            "report_path": dict(result.get("named_outputs", {})).get("daily_report", {}).get("report_path"),
        },
        indent=2,
        sort_keys=True,
    )


def _render_ibkr_snapshot(snapshot: dict[str, Any]) -> str:
    option_chain = dict(snapshot.get("option_chain", {}))
    return json.dumps(
        {
            "as_of": snapshot.get("as_of"),
            "source": snapshot.get("source"),
            "symbols": snapshot.get("symbols"),
            "option_contract_count": len(option_chain.get("option_quotes", [])),
            "expirations": option_chain.get("expirations", []),
            "strikes": option_chain.get("strikes", []),
            "provider_metadata": snapshot.get("provider_metadata", {}),
        },
        indent=2,
        sort_keys=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the volatility regime application.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run-daily", help="Run the daily regime workflow.")
    run_parser.add_argument("--input", required=True, help="Path to a JSON input payload.")
    run_parser.add_argument("--agent", default=str(default_agent_path()), help="Path to the agent YAML.")
    run_parser.add_argument("--storage-root", default=None)
    run_parser.add_argument("--database-url", default=None)
    run_parser.add_argument("--langsmith-tracing", action="store_true")
    run_parser.add_argument("--langsmith-project", default=None)
    run_parser.add_argument("--as-of-date", default=None, help="Optional historical YYYY-MM-DD rewind date.")
    run_parser.add_argument("--output", choices=("summary", "report", "internal"), default="summary")

    resume_parser = subparsers.add_parser("resume", help="Resume a review-gated run.")
    resume_parser.add_argument("--run-id", required=True)
    resume_parser.add_argument("--decision", choices=("approved", "rejected"), required=True)
    resume_parser.add_argument("--notes", default=None)
    resume_parser.add_argument("--agent", default=str(default_agent_path()), help="Path to the agent YAML.")
    resume_parser.add_argument("--storage-root", default=None)
    resume_parser.add_argument("--database-url", default=None)
    resume_parser.add_argument("--langsmith-tracing", action="store_true")
    resume_parser.add_argument("--langsmith-project", default=None)
    resume_parser.add_argument("--output", choices=("summary", "report", "internal"), default="summary")

    ibkr_parser = subparsers.add_parser(
        "fetch-ibkr-snapshot",
        help="Fetch an IBKR-backed SPY underlying + option-chain snapshot.",
    )
    ibkr_parser.add_argument("--symbol", default="SPY")
    ibkr_parser.add_argument("--host", default="127.0.0.1")
    ibkr_parser.add_argument("--port", type=int, default=7497)
    ibkr_parser.add_argument("--client-id", type=int, default=73)
    ibkr_parser.add_argument("--market-data-type", type=int, default=1)
    ibkr_parser.add_argument("--exchange", default="SMART")
    ibkr_parser.add_argument("--option-exchange", default="SMART")
    ibkr_parser.add_argument("--currency", default="USD")
    ibkr_parser.add_argument("--expiry-count", type=int, default=2)
    ibkr_parser.add_argument("--strike-count", type=int, default=8)
    ibkr_parser.add_argument("--min-days-to-expiry", type=int, default=0)
    ibkr_parser.add_argument("--rights", nargs="+", default=["C", "P"])
    ibkr_parser.add_argument("--expirations", nargs="*", default=[])
    ibkr_parser.add_argument("--strikes", nargs="*", type=float, default=[])
    ibkr_parser.add_argument("--output", default=None, help="Optional path to write the snapshot JSON.")

    args = parser.parse_args()

    if args.command == "run-daily":
        input_payload = _load_input(args.input)
        if args.as_of_date:
            input_payload["as_of_date"] = str(args.as_of_date)
        result = run_daily_regime_agent(
            input_payload=input_payload,
            agent_path=args.agent,
            storage_root=args.storage_root,
            database_url=args.database_url,
            langsmith_tracing=args.langsmith_tracing,
            langsmith_project=args.langsmith_project,
        )
        print(_render_output(result, args.output))
        return

    if args.command == "fetch-ibkr-snapshot":
        pipe = IBKRDataPipe(
            connection=IBKRConnectionConfig(
                host=args.host,
                port=args.port,
                client_id=args.client_id,
                market_data_type=args.market_data_type,
            )
        )
        snapshot = pipe.fetch_market_snapshot(
            IBKROptionChainRequest(
                symbol=args.symbol,
                exchange=args.exchange,
                currency=args.currency,
                option_exchange=args.option_exchange,
                rights=tuple(str(item).upper() for item in args.rights),
                expiry_count=args.expiry_count,
                strike_count=args.strike_count,
                expirations=tuple(str(item) for item in args.expirations),
                strikes=tuple(float(item) for item in args.strikes),
                min_days_to_expiry=args.min_days_to_expiry,
            )
        ).to_dict()
        if args.output:
            output_path = Path(args.output).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
        print(_render_ibkr_snapshot(snapshot))
        return

    result = resume_daily_regime_run(
        run_id=args.run_id,
        decision=args.decision,
        notes=args.notes,
        agent_path=args.agent,
        storage_root=args.storage_root,
        database_url=args.database_url,
        langsmith_tracing=args.langsmith_tracing,
        langsmith_project=args.langsmith_project,
    )
    print(_render_output(result, args.output))


if __name__ == "__main__":
    main()
