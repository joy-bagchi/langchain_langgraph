# Vol Visualizer

`vol_visualizer` is read-only. It discovers the immutable, dated Parquet
datasets published by `vol_surface_publisher`, validates their checksums, and
creates an interactive Plotly cube: strike × DTE × observation date, colored
by implied volatility.

Install optional runtime dependencies:

```powershell
pip install ib-insync google-cloud-storage pyarrow plotly
```

With TWS or IB Gateway running and ADC configured:

```powershell
python -m vol_visualizer.cli render --output iv_cube.html
```

The default GCS namespace is
`gs://marketphysics-market-manifold-data/market-manifold/option-chain-iv`.
