from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd

from src.data.yahoo_eod_provider import _fetch_yfinance_frame, fetch_raw_yahoo_chart_frame


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


def test_fetch_raw_yahoo_chart_frame_parses_chart_payload(monkeypatch) -> None:
    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [1704067200, 1704153600],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0, 101.0],
                                "high": [102.0, 103.0],
                                "low": [99.0, 100.5],
                                "close": [101.0, 102.5],
                                "volume": [1000, 1100],
                            }
                        ],
                        "adjclose": [
                            {
                                "adjclose": [100.8, 102.2],
                            }
                        ],
                    },
                }
            ],
            "error": None,
        }
    }

    def _fake_get(url, params, headers, timeout):
        assert "SPY" in url
        assert headers["User-Agent"] == "Mozilla/5.0"
        assert params["interval"] == "1d"
        return _FakeResponse(payload)

    monkeypatch.setattr("requests.get", _fake_get)
    frame = fetch_raw_yahoo_chart_frame(symbol="SPY", start_date="2024-01-01", end_date="2024-01-03")

    assert len(frame) == 2
    assert list(frame.columns) == ["date", "open", "high", "low", "close", "adj_close", "volume", "source"]
    assert str(frame.iloc[0]["date"]) == "2024-01-01"
    assert float(frame.iloc[0]["adj_close"]) == 100.8
    assert str(frame.iloc[0]["source"]) == "yahoo_chart"


def test_fetch_yfinance_frame_handles_multiindex_columns(monkeypatch) -> None:
    index = pd.to_datetime(["2024-01-01", "2024-01-02"])
    frame = pd.DataFrame(
        {
            ("Open", "SPY"): [100.0, 101.0],
            ("High", "SPY"): [101.0, 102.0],
            ("Low", "SPY"): [99.0, 100.0],
            ("Close", "SPY"): [100.5, 101.5],
            ("Adj Close", "SPY"): [100.4, 101.4],
            ("Volume", "SPY"): [1000, 1100],
        },
        index=index,
    )
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)

    class _FakeTicker:
        def history(self, **kwargs):
            return pd.DataFrame()

    fake_yf = SimpleNamespace(
        download=lambda **kwargs: frame,
        Ticker=lambda symbol: _FakeTicker(),
        set_tz_cache_location=lambda _: None,
    )
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)

    normalized = _fetch_yfinance_frame(symbol="SPY", start_date="2024-01-01", end_date="2024-01-03")

    assert len(normalized) == 2
    assert list(normalized.columns) == ["date", "open", "high", "low", "close", "adj_close", "volume", "source"]
    assert float(normalized.iloc[0]["close"]) == 100.5
    assert str(normalized.iloc[0]["source"]) == "yfinance"
