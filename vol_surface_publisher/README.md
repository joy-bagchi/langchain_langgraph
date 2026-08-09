# Vol Surface Publisher

This independent producer fetches one live IBKR option chain, extracts the
positive implied-volatility quotes, and writes an immutable Parquet dataset for
that observation time. Each completed dataset is appended to the protected GCS
catalog at `manifests/catalog.json`; the visualizer uses that catalog to load
multiple dates.

The publisher captures one surface per completed U.S. equity session, using
IBKR frozen market data after 16:15 America/New_York. A weekend or holiday run
therefore tries the previous completed session; the GCS catalog prevents a
duplicate if that session is already published. A run before the close returns
`skipped_waiting_for_market_close`. It cannot reconstruct historical IV chains
for dates IBKR does not retain; those gaps remain visible rather than being
filled with relabeled current data.

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
