# Vol Surface Publisher

This independent producer fetches one live IBKR option chain, extracts the
positive implied-volatility quotes, and writes an immutable Parquet dataset for
that observation time. Each completed dataset is appended to the protected GCS
catalog at `manifests/catalog.json`; the visualizer uses that catalog to load
multiple dates.

The IBKR request is a live observation. Run it on the observation dates you
need (for example, after the market close) to accumulate the cube history; it
does not claim to reconstruct historical option IV from daily VIX/VVIX values.

```powershell
pip install ib-insync google-cloud-storage pyarrow
python -m vol_surface_publisher.cli --symbol SPY --port 4001 --dry-run
python -m vol_surface_publisher.cli --symbol SPY --port 4001
```

To run the market-price, volatility-regime, and option-surface publishers in
one sequence, use the colocated launchers:

```powershell
.\vol_surface_publisher\run_all_publishers.ps1 -DryRun
.\vol_surface_publisher\run_all_publishers.ps1
```

From CMD, run `vol_surface_publisher\run_all_publishers.bat`. Both launchers
change to the repository root before invoking their Python modules.
