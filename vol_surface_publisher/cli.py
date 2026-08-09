"""CLI for the option-surface publisher."""
from __future__ import annotations
import argparse, json
from .publisher import DEFAULT_BUCKET, DEFAULT_PREFIX, collect_and_publish
def main() -> None:
    parser = argparse.ArgumentParser(description="Collect and publish a dated IBKR option IV surface.")
    parser.add_argument("--symbol", default="SPY"); parser.add_argument("--host", default="127.0.0.1"); parser.add_argument("--port", type=int, default=4001); parser.add_argument("--client-id", type=int, default=74)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET); parser.add_argument("--prefix", default=DEFAULT_PREFIX); parser.add_argument("--project"); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(); print(json.dumps(collect_and_publish(symbol=args.symbol, host=args.host, port=args.port, client_id=args.client_id, bucket=args.bucket, prefix=args.prefix, project=args.project, dry_run=args.dry_run), indent=2))
if __name__ == "__main__": main()
