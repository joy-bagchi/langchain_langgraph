import json

from vol_visualizer.reader import load_surface_catalog


class FakeStorage:
    def download_bytes(self, bucket: str, object_name: str) -> bytes:
        assert bucket == "test-bucket"
        assert object_name == "market-manifold/option-chain-iv/manifests/catalog.json"
        return json.dumps({
            "manifest_schema_version": "option_chain_iv_catalog.v1",
            "datasets": [
                {"dataset_id": "second", "observation_date": "2026-08-08", "observation_time": "2026-08-08T20:15:00Z", "symbols": ["SPY"]},
                {"dataset_id": "first", "observation_date": "2026-08-07", "observation_time": "2026-08-07T20:15:00Z", "symbols": ["SPY"]},
            ],
        }).encode("utf-8")


def test_load_surface_catalog_orders_entries_without_parquet_downloads() -> None:
    catalog = load_surface_catalog(bucket="test-bucket", storage_client=FakeStorage())
    assert catalog.dataset_id.tolist() == ["first", "second"]
