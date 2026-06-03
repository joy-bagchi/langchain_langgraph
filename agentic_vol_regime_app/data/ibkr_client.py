"""IBKR-backed market data pipe for underlying quotes and option chains."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Protocol

from agentic_vol_regime_app.contracts import (
    ObservationRecord,
    OptionChainSnapshot,
    OptionGreekRecord,
    OptionQuoteRecord,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _first_number(*values: Any) -> float | None:
    for value in values:
        number = _safe_float(value)
        if number is not None:
            return number
    return None


@dataclass(frozen=True, slots=True)
class IBKRConnectionConfig:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 73
    readonly: bool = True
    timeout_seconds: float = 10.0
    market_data_type: int = 1


@dataclass(frozen=True, slots=True)
class IBKROptionChainRequest:
    symbol: str = "SPY"
    exchange: str = "SMART"
    currency: str = "USD"
    option_exchange: str = "SMART"
    rights: tuple[str, ...] = ("C", "P")
    expiry_count: int = 2
    strike_count: int = 8
    expirations: tuple[str, ...] = ()
    strikes: tuple[float, ...] = ()
    min_days_to_expiry: int = 0

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "IBKROptionChainRequest":
        rights = payload.get("rights", ("C", "P"))
        expirations = payload.get("expirations", ())
        strikes = payload.get("strikes", ())
        return cls(
            symbol=str(payload.get("symbol", "SPY")),
            exchange=str(payload.get("exchange", "SMART")),
            currency=str(payload.get("currency", "USD")),
            option_exchange=str(payload.get("option_exchange", "SMART")),
            rights=tuple(str(item).upper() for item in rights),
            expiry_count=int(payload.get("expiry_count", 2)),
            strike_count=int(payload.get("strike_count", 8)),
            expirations=tuple(str(item) for item in expirations),
            strikes=tuple(float(item) for item in strikes),
            min_days_to_expiry=int(payload.get("min_days_to_expiry", 0)),
        )


class IBKRMarketDataClient(Protocol):
    """Low-level market data client contract."""

    def fetch_market_snapshot(self, request: IBKROptionChainRequest) -> dict[str, Any]:
        """Fetch an underlying quote and selected option-chain data."""


class IBKRLiveClient:
    """Real IBKR client backed by ib_insync."""

    def __init__(self, connection: IBKRConnectionConfig) -> None:
        self.connection = connection

    def fetch_market_snapshot(self, request: IBKROptionChainRequest) -> dict[str, Any]:
        try:
            from ib_insync import IB, Option, Stock
        except ImportError as exc:
            raise RuntimeError(
                "IBKR live market data requires the optional 'ib-insync' package."
            ) from exc

        ib = IB()
        ib.connect(
            self.connection.host,
            self.connection.port,
            clientId=self.connection.client_id,
            readonly=self.connection.readonly,
            timeout=self.connection.timeout_seconds,
        )
        try:
            ib.reqMarketDataType(self.connection.market_data_type)

            stock_contract = Stock(request.symbol, request.exchange, request.currency)
            qualified_stock = ib.qualifyContracts(stock_contract)
            if not qualified_stock:
                raise RuntimeError(f"Unable to qualify IBKR stock contract for {request.symbol}.")
            stock_contract = qualified_stock[0]

            underlying_ticker = ib.reqTickers(stock_contract)[0]
            underlying_price = _first_number(
                underlying_ticker.marketPrice(),
                underlying_ticker.last,
                underlying_ticker.close,
                underlying_ticker.bid,
                underlying_ticker.ask,
            )
            if underlying_price is None:
                raise RuntimeError(f"IBKR returned no usable underlying price for {request.symbol}.")

            option_parameters = ib.reqSecDefOptParams(
                request.symbol,
                "",
                stock_contract.secType,
                stock_contract.conId,
            )
            if not option_parameters:
                raise RuntimeError(f"IBKR returned no option parameters for {request.symbol}.")

            selected_parameters = self._select_option_parameters(
                option_parameters,
                preferred_exchange=request.option_exchange,
            )
            expirations = self._select_expirations(
                available=selected_parameters.expirations,
                requested=request.expirations,
                limit=request.expiry_count,
                min_days_to_expiry=request.min_days_to_expiry,
            )
            strikes = self._select_strikes(
                available=selected_parameters.strikes,
                requested=request.strikes,
                underlying_price=underlying_price,
                limit=request.strike_count,
            )
            option_contracts = []
            for expiry in expirations:
                for strike in strikes:
                    for right in request.rights:
                        option_contracts.append(
                            Option(
                                request.symbol,
                                expiry,
                                strike,
                                right,
                                request.option_exchange,
                                multiplier=str(getattr(selected_parameters, "multiplier", "") or ""),
                                currency=request.currency,
                                tradingClass=getattr(selected_parameters, "tradingClass", None),
                            )
                        )
            if not option_contracts:
                raise RuntimeError("No option contracts were selected from the IBKR chain parameters.")

            qualified_option_contracts = ib.qualifyContracts(*option_contracts)
            tickers = list(ib.reqTickers(*qualified_option_contracts))
            option_quotes = [
                self._normalize_option_ticker(ticker)
                for ticker in tickers
            ]

            return {
                "schema_version": "observation.v1",
                "as_of": _utc_now(),
                "source": "IBKR",
                "symbols": {
                    request.symbol: {
                        "last": underlying_price,
                        "close": _safe_float(underlying_ticker.close),
                        "bid": _safe_float(underlying_ticker.bid),
                        "ask": _safe_float(underlying_ticker.ask),
                        "volume": _safe_float(underlying_ticker.volume),
                    }
                },
                "history": {},
                "quality": {
                    "is_complete": True,
                    "warnings": [],
                    "stale_fields": [],
                },
                "option_chain": OptionChainSnapshot(
                    underlying_symbol=request.symbol,
                    underlying_price=underlying_price,
                    fetched_at=_utc_now(),
                    exchange=request.option_exchange,
                    currency=request.currency,
                    expirations=list(expirations),
                    strikes=[float(strike) for strike in strikes],
                    rights=list(request.rights),
                    option_quotes=[quote.to_dict() for quote in option_quotes],
                ).to_dict(),
                "provider_metadata": {
                    "host": self.connection.host,
                    "port": self.connection.port,
                    "client_id": self.connection.client_id,
                    "market_data_type": self.connection.market_data_type,
                    "qualified_option_contract_count": len(qualified_option_contracts),
                },
            }
        finally:
            ib.disconnect()

    @staticmethod
    def _select_option_parameters(parameters: list[Any], preferred_exchange: str) -> Any:
        preferred_upper = preferred_exchange.upper()
        ranked = sorted(
            parameters,
            key=lambda item: (
                0 if str(getattr(item, "exchange", "")).upper() == preferred_upper else 1,
                -len(getattr(item, "strikes", []) or []),
                -len(getattr(item, "expirations", []) or []),
            ),
        )
        return ranked[0]

    @staticmethod
    def _select_expirations(
        *,
        available: list[str] | tuple[str, ...],
        requested: tuple[str, ...],
        limit: int,
        min_days_to_expiry: int,
    ) -> list[str]:
        today = datetime.now(timezone.utc).date()
        available_sorted = sorted(str(item) for item in available)
        if requested:
            selected = [expiry for expiry in available_sorted if expiry in requested]
        else:
            selected = []
            for expiry in available_sorted:
                try:
                    expiry_date = datetime.strptime(expiry, "%Y%m%d").date()
                except ValueError:
                    continue
                if (expiry_date - today).days >= min_days_to_expiry:
                    selected.append(expiry)
                if len(selected) >= limit:
                    break
        return selected[:limit]

    @staticmethod
    def _select_strikes(
        *,
        available: list[float] | tuple[float, ...],
        requested: tuple[float, ...],
        underlying_price: float,
        limit: int,
    ) -> list[float]:
        if requested:
            requested_set = {float(item) for item in requested}
            return [float(strike) for strike in available if float(strike) in requested_set]
        sorted_strikes = sorted(float(strike) for strike in available)
        ranked = sorted(sorted_strikes, key=lambda strike: (abs(strike - underlying_price), strike))
        selected = sorted(ranked[:limit])
        return selected

    @staticmethod
    def _normalize_option_ticker(ticker: Any) -> OptionQuoteRecord:
        contract = ticker.contract
        right = str(getattr(contract, "right", "")).upper()
        greeks_source = (
            getattr(ticker, "modelGreeks", None)
            or getattr(ticker, "lastGreeks", None)
            or getattr(ticker, "bidGreeks", None)
            or getattr(ticker, "askGreeks", None)
        )
        greeks = OptionGreekRecord(
            delta=_safe_float(getattr(greeks_source, "delta", None)),
            gamma=_safe_float(getattr(greeks_source, "gamma", None)),
            theta=_safe_float(getattr(greeks_source, "theta", None)),
            vega=_safe_float(getattr(greeks_source, "vega", None)),
            implied_vol=_safe_float(
                getattr(greeks_source, "impliedVol", None) or getattr(ticker, "impliedVolatility", None)
            ),
            opt_price=_safe_float(getattr(greeks_source, "optPrice", None)),
            pv_dividend=_safe_float(getattr(greeks_source, "pvDividend", None)),
            und_price=_safe_float(getattr(greeks_source, "undPrice", None)),
        )
        mark = None
        bid = _safe_float(getattr(ticker, "bid", None))
        ask = _safe_float(getattr(ticker, "ask", None))
        if bid is not None and ask is not None:
            mark = round((bid + ask) / 2.0, 6)
        else:
            mark = _first_number(getattr(ticker, "last", None), getattr(ticker, "close", None))
        open_interest = _safe_float(
            getattr(ticker, "callOpenInterest" if right == "C" else "putOpenInterest", None)
        )
        return OptionQuoteRecord(
            symbol=str(getattr(contract, "localSymbol", None) or getattr(contract, "symbol", "")),
            expiry=str(getattr(contract, "lastTradeDateOrContractMonth", "")),
            strike=float(getattr(contract, "strike", 0.0)),
            right=right,
            exchange=str(getattr(contract, "exchange", "")),
            currency=str(getattr(contract, "currency", "")),
            bid=bid,
            ask=ask,
            last=_safe_float(getattr(ticker, "last", None)),
            close=_safe_float(getattr(ticker, "close", None)),
            mark=mark,
            volume=_safe_float(getattr(ticker, "volume", None)),
            open_interest=open_interest,
            bid_size=_safe_float(getattr(ticker, "bidSize", None)),
            ask_size=_safe_float(getattr(ticker, "askSize", None)),
            last_size=_safe_float(getattr(ticker, "lastSize", None)),
            multiplier=str(getattr(contract, "multiplier", "") or ""),
            greeks=greeks.to_dict(),
        )


@dataclass(slots=True)
class IBKRDataPipe:
    """Stable high-level data pipe over the lower-level IBKR client."""

    connection: IBKRConnectionConfig = field(default_factory=IBKRConnectionConfig)
    client_factory: Callable[[IBKRConnectionConfig], IBKRMarketDataClient] = IBKRLiveClient

    def fetch_market_snapshot(
        self,
        request: IBKROptionChainRequest,
    ) -> ObservationRecord:
        client = self.client_factory(self.connection)
        payload = client.fetch_market_snapshot(request)
        return ObservationRecord(
            schema_version=str(payload.get("schema_version", "observation.v1")),
            as_of=str(payload["as_of"]),
            source=str(payload.get("source", "IBKR")),
            symbols=dict(payload.get("symbols", {})),
            history={key: list(value) for key, value in dict(payload.get("history", {})).items()},
            quality=dict(payload.get("quality", {})),
            option_chain=dict(payload.get("option_chain", {})),
            provider_metadata=dict(payload.get("provider_metadata", {})),
        )
