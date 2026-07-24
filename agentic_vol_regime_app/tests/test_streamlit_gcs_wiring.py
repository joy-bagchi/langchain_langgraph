from __future__ import annotations

import streamlit_app


def test_feature_store_gcs_sync_forwards_shared_ibkr_settings(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class Result:
        def to_dict(self) -> dict[str, object]:
            return {"gcs_publish": {"status": "published"}}

    def fake_sync(**kwargs):
        calls.update(kwargs)
        return Result()

    monkeypatch.setattr(streamlit_app, "sync_and_publish_vol_regime_history", fake_sync)

    result = streamlit_app._sync_feature_store_vol_regime_history(
        history_days=2520,
        as_of_date="2026-07-24",
        host="127.0.0.1",
        port=4001,
        client_id=73,
        market_data_type=1,
    )

    assert result["gcs_publish"]["status"] == "published"
    assert calls == {
        "history_days": 2520,
        "target_end_date": "2026-07-24",
        "host": "127.0.0.1",
        "port": 4001,
        "client_id": 73,
        "market_data_type": 1,
    }
