"""Command-line entry point for market-physics-core."""
from __future__ import annotations
import argparse
from collections.abc import Sequence
from datetime import date
import json
from pathlib import Path
import subprocess
import tomllib
from market_physics_core import __version__
from market_physics_core.adapters.baseline_store.gcs import GCSBaselineStore
from market_physics_core.adapters.baseline_store.local import write_local_history
from market_physics_core.adapters.market_data.gcs import GCSMarketDataReader
from market_physics_core.applications.skew_baseline.weekly import calculate_weekly

def build_parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(prog="market-physics-core",description="Deterministic analysis engine for Market Physics.")
    parser.add_argument("--version",action="version",version=f"%(prog)s {__version__}")
    commands=parser.add_subparsers(dest="command")
    baseline=commands.add_parser("skew-baseline",help="Calculate, publish, and retrieve skew baselines.")
    baseline.add_argument("action",nargs="?",default="run",choices=("run","publish","latest","history","backfill"))
    baseline.add_argument("--config",type=Path,default=Path("configs/skew_baseline.example.toml"))
    baseline.add_argument("--output-root",type=Path)
    baseline.add_argument("--as-of",type=date.fromisoformat)
    baseline.add_argument("--start",type=date.fromisoformat)
    baseline.add_argument("--end",type=date.fromisoformat)
    baseline.add_argument("--include-revisions",action="store_true")
    baseline.add_argument("--retrospective",action="store_true")
    baseline.add_argument("--no-advance-latest",action="store_true")
    return parser

def _config(path: Path):
    with path.open("rb") as stream: return tomllib.load(stream)

def _store(config):
    output=config["outputs"]
    return GCSBaselineStore(config["inputs"]["gcp_project"],output["gcs_results_bucket"],output["gcs_results_prefix"])

def _inputs(config):
    value=config["inputs"]
    return GCSMarketDataReader(project=value["gcp_project"],bucket=value["gcs_bucket"],option_catalog_object=value["option_catalog_object"],volatility_history_manifest_object=value["volatility_history_manifest_object"]).load()

def _commit():
    return subprocess.run(["git","rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip()

def _calculate(config,args,as_of=None):
    return calculate_weekly(_inputs(config),code_commit=_commit(),as_of=as_of or args.as_of,
        sigma_multiplier=float(config["sigma_multiplier"]),methodology_version=str(config["methodology_version"]),
        rolling_window_sessions=int(config["rolling_window_sessions"]),calendar_name=str(config["exchange_calendar"]),
        minimum_usable_count=int(config["operations"]["minimum_usable_count"]),retrospective=args.retrospective)

def main(argv: Sequence[str]|None=None)->int:
    args=build_parser().parse_args(argv)
    if args.command!="skew-baseline": return 0
    config=_config(args.config)
    if args.action=="latest": print(json.dumps(_store(config).latest(),indent=2,sort_keys=True)); return 0
    if args.action=="history": print(json.dumps(_store(config).history(include_revisions=args.include_revisions),indent=2,sort_keys=True)); return 0
    if args.action=="backfill":
        if not args.start or not args.end or args.start>args.end: raise SystemExit("backfill requires --start and --end in ascending order")
        inputs=_inputs(config); weeks=sorted({date.fromisoformat(s.observation_date).isocalendar()[:2] for s in inputs.surfaces if args.start<=date.fromisoformat(s.observation_date)<=args.end})
        results=[]
        for year,week in weeks:
            candidates=[date.fromisoformat(s.observation_date) for s in inputs.surfaces if date.fromisoformat(s.observation_date).isocalendar()[:2]==(year,week) and args.start<=date.fromisoformat(s.observation_date)<=args.end]
            summary,rows=calculate_weekly(inputs,code_commit=_commit(),as_of=max(candidates),sigma_multiplier=float(config["sigma_multiplier"]),methodology_version=str(config["methodology_version"]),rolling_window_sessions=int(config["rolling_window_sessions"]),calendar_name=str(config["exchange_calendar"]),minimum_usable_count=int(config["operations"]["minimum_usable_count"]),retrospective=True)
            results.append(_store(config).publish(summary,rows,publication_kind="retrospective",advance_latest=False))
        print(json.dumps(results,indent=2,sort_keys=True)); return 0
    summary,rows=_calculate(config,args)
    root=args.output_root or Path(config["outputs"]["local_history_path"])
    run_dir,checksums=write_local_history(root,summary,rows)
    result={"run_directory":str(run_dir.resolve()),"checksums":checksums,"summary":summary}
    if args.action=="publish": result["publication"]=_store(config).publish(summary,rows,publication_kind=summary["calculation_context"],advance_latest=not args.no_advance_latest)
    print(json.dumps(result,indent=2,sort_keys=True)); return 0

if __name__=="__main__": raise SystemExit(main())
