from __future__ import annotations

from datetime import date

from src.data.historical_source_router import stitch_history_with_yahoo_fallback
from src.data.yahoo_eod_provider import YahooEODFetchResult, YahooEODPoint


def test_yahoo_fallback_stitches_and_prefers_ibkr_on_overlap() -> None:
    ibkr_values = [200.0, 201.0, 202.0, 203.0, 204.0]  # inferred dates: 2020-01-06..2020-01-10

    def _fake_yahoo_fetcher(*, symbol: str, start_date: str, end_date: str):
        assert symbol == "XLK"
        assert start_date == "2020-01-01"
        return [
            YahooEODPoint(day=date(2020, 1, 1), close=180.0),
            YahooEODPoint(day=date(2020, 1, 2), close=181.0),
            YahooEODPoint(day=date(2020, 1, 3), close=182.0),
            YahooEODPoint(day=date(2020, 1, 6), close=999.0),  # overlap: should lose to IBKR
        ]

    stitched = stitch_history_with_yahoo_fallback(
        symbol="XLK",
        ibkr_values=ibkr_values,
        as_of_date="2020-01-10",
        preferred_start_date="2020-01-01",
        minimum_required_start_date="2020-01-01",
        yahoo_fetcher=_fake_yahoo_fetcher,
    )

    assert stitched.coverage["status"] == "ok"
    assert stitched.coverage["source_mix"] == "mixed"
    assert stitched.coverage["final_earliest_date"] == "2020-01-01"
    assert stitched.coverage["row_count"] == len(stitched.values)
    # 2020-01-06 close should come from IBKR, not Yahoo's overlapping 999.0.
    assert 999.0 not in stitched.values


def test_yahoo_fallback_reports_insufficient_when_coverage_is_still_too_shallow() -> None:
    ibkr_values = [300.0, 301.0]  # inferred late start near as_of

    def _empty_yahoo_fetcher(*, symbol: str, start_date: str, end_date: str):
        return []

    stitched = stitch_history_with_yahoo_fallback(
        symbol="XLF",
        ibkr_values=ibkr_values,
        as_of_date="2020-01-10",
        preferred_start_date="2020-01-01",
        minimum_required_start_date="2020-01-01",
        yahoo_fetcher=_empty_yahoo_fetcher,
    )

    assert stitched.coverage["status"] == "insufficient"
    assert stitched.coverage["final_earliest_date"] > "2020-01-01"


def test_raw_yahoo_provider_metadata_is_preserved_when_yfinance_fails() -> None:
    def _metadata_fetcher(*, symbol: str, start_date: str, end_date: str):
        assert symbol == "SPY"
        return YahooEODFetchResult(
            points=[
                YahooEODPoint(day=date(2020, 1, 1), close=100.0, source="yahoo_chart"),
                YahooEODPoint(day=date(2020, 1, 2), close=101.0, source="yahoo_chart"),
            ],
            provider_used="yahoo_chart",
            yfinance_adapter_failure=True,
            external_provider_available=True,
            warnings=["yfinance adapter failed"],
        )

    stitched = stitch_history_with_yahoo_fallback(
        symbol="SPY",
        ibkr_values=[],
        as_of_date="2020-01-10",
        preferred_start_date="2020-01-01",
        minimum_required_start_date="2020-01-01",
        yahoo_fetcher_with_metadata=_metadata_fetcher,
    )

    assert stitched.coverage["yahoo_provider_used"] == "yahoo_chart"
    assert stitched.coverage["yfinance_adapter_failure"] is True
    assert stitched.coverage["external_provider_available"] is True
