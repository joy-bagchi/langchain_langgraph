"""IBKR-backed market data pipe for underlying quotes and option chains."""

from __future__ import annotations

import asyncio
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


DEFAULT_VOL_REGIME_SYMBOLS = ("SPY", "VIX", "VVIX", "VIX9D", "VIX3M", "VIX6M", "VIX9M")
_IBKR_UNSET_DOUBLE = 1.7976931348623157e308
_IBKR_UNSET_INT = 2147483647


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    if number in {-1.0, -2.0, _IBKR_UNSET_DOUBLE, float(_IBKR_UNSET_INT)}:
        return None
    return number


def _first_number(*values: Any) -> float | None:
    for value in values:
        number = _safe_float(value)
        if number is not None:
            return number
    return None


def _ensure_thread_event_loop() -> asyncio.AbstractEventLoop:
    """Create a thread-local asyncio loop when the caller thread lacks one."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        pass

    policy = asyncio.get_event_loop_policy()
    try:
        return policy.get_event_loop()
    except RuntimeError:
        loop = policy.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


@dataclass(frozen=True, slots=True)
class IBKRConnectionConfig:
    host: str = "127.0.0.1"
    port: int = 4001
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


@dataclass(frozen=True, slots=True)
class IBKRVolRegimeSnapshotRequest:
    option_chain: IBKROptionChainRequest = field(default_factory=IBKROptionChainRequest)
    history_days: int = 30
    regime_symbols: tuple[str, ...] = DEFAULT_VOL_REGIME_SYMBOLS
    index_exchange: str = "CBOE"
    currency: str = "USD"

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "IBKRVolRegimeSnapshotRequest":
        option_chain = IBKROptionChainRequest.from_payload(payload)
        configured_symbols = payload.get("regime_symbols", DEFAULT_VOL_REGIME_SYMBOLS)
        normalized_symbols = tuple(
            dict.fromkeys(str(item).upper() for item in configured_symbols if str(item).strip())
        )
        if option_chain.symbol.upper() not in normalized_symbols:
            normalized_symbols = (option_chain.symbol.upper(), *normalized_symbols)
        return cls(
            option_chain=option_chain,
            history_days=max(int(payload.get("history_days", 30)), 5),
            regime_symbols=normalized_symbols or DEFAULT_VOL_REGIME_SYMBOLS,
            index_exchange=str(payload.get("index_exchange", "CBOE")),
            currency=str(payload.get("currency", option_chain.currency or "USD")),
        )


class IBKRMarketDataClient(Protocol):
    """Low-level market data client contract."""

    def fetch_market_snapshot(self, request: IBKROptionChainRequest) -> dict[str, Any]:
        """Fetch an underlying quote and selected option-chain data."""

    def fetch_vol_regime_snapshot(self, request: IBKRVolRegimeSnapshotRequest) -> dict[str, Any]:
        """Fetch the full daily volatility-regime observation payload."""


class IBKRLiveClient:
    """Real IBKR client backed by ib_insync."""

    def __init__(self, connection: IBKRConnectionConfig) -> None:
        self.connection = connection

    def fetch_market_snapshot(self, request: IBKROptionChainRequest) -> dict[str, Any]:
        ib = self._connect()
        try:
            stock_contract = self._qualify_stock_contract(
                ib,
                symbol=request.symbol,
                exchange=request.exchange,
                currency=request.currency,
            )
            underlying_quote, underlying_price = self._request_quote_payload(ib, stock_contract)
            if underlying_price is None:
                raise RuntimeError(f"IBKR returned no usable underlying price for {request.symbol}.")
            option_chain = self._fetch_option_chain_snapshot(
                ib,
                stock_contract=stock_contract,
                request=request,
                underlying_price=underlying_price,
            )

            return {
                "schema_version": "observation.v1",
                "as_of": _utc_now(),
                "source": "IBKR",
                "symbols": {request.symbol: underlying_quote},
                "history": {},
                "quality": {
                    "is_complete": True,
                    "warnings": [],
                    "stale_fields": [],
                },
                "option_chain": option_chain,
                "provider_metadata": {
                    "host": self.connection.host,
                    "port": self.connection.port,
                    "client_id": self.connection.client_id,
                    "market_data_type": self.connection.market_data_type,
                    "qualified_option_contract_count": len(option_chain.get("option_quotes", [])),
                },
            }
        finally:
            ib.disconnect()

    def fetch_vol_regime_snapshot(self, request: IBKRVolRegimeSnapshotRequest) -> dict[str, Any]:
        ib = self._connect()
        try:
            option_request = request.option_chain
            stock_contract = self._qualify_stock_contract(
                ib,
                symbol=option_request.symbol,
                exchange=option_request.exchange,
                currency=option_request.currency,
            )
            underlying_quote, underlying_price = self._request_quote_payload(ib, stock_contract)
            if underlying_price is None:
                raise RuntimeError(
                    f"IBKR returned no usable underlying price for {option_request.symbol}."
                )
            option_chain = self._fetch_option_chain_snapshot(
                ib,
                stock_contract=stock_contract,
                request=option_request,
                underlying_price=underlying_price,
            )

            symbols_payload: dict[str, dict[str, Any]] = {
                option_request.symbol.upper(): underlying_quote,
            }
            history_payload: dict[str, list[float]] = {}
            warnings: list[str] = []
            stale_fields: list[str] = []

            for symbol in request.regime_symbols:
                normalized_symbol = symbol.upper()
                if normalized_symbol == option_request.symbol.upper():
                    contract = stock_contract
                    quote_payload = underlying_quote
                else:
                    contract = self._qualify_index_contract(
                        ib,
                        symbol=normalized_symbol,
                        exchange=request.index_exchange,
                        currency=request.currency,
                    )
                    if contract is None:
                        warnings.append(f"unable to qualify contract for {normalized_symbol}")
                        continue
                    quote_payload, _ = self._request_quote_payload(ib, contract)

                last_value = quote_payload.get("last")
                if last_value in {None, ""}:
                    stale_fields.append(f"{normalized_symbol}.last")
                symbols_payload[normalized_symbol] = quote_payload

                history_key = (
                    f"{normalized_symbol}_close"
                    if normalized_symbol == option_request.symbol.upper()
                    else normalized_symbol
                )
                history_values = self._request_daily_history(
                    ib,
                    contract,
                    history_days=request.history_days,
                )
                if history_values:
                    history_payload[history_key] = history_values
                else:
                    warnings.append(f"unable to load daily history for {history_key}")

            missing_symbols = [
                symbol for symbol in request.regime_symbols if symbol.upper() not in symbols_payload
            ]
            required_history = [
                ("SPY_close" if symbol.upper() == option_request.symbol.upper() else symbol.upper())
                for symbol in request.regime_symbols
            ]
            missing_history = [
                key for key in required_history if len(history_payload.get(key, [])) < min(request.history_days, 22)
            ]
            quality = {
                "is_complete": not missing_symbols and not missing_history and not stale_fields,
                "warnings": warnings,
                "stale_fields": stale_fields,
                "missing_symbols": missing_symbols,
                "missing_history": missing_history,
            }

            return {
                "schema_version": "observation.v1",
                "as_of": _utc_now(),
                "source": "IBKR",
                "symbols": symbols_payload,
                "history": history_payload,
                "quality": quality,
                "option_chain": option_chain,
                "provider_metadata": {
                    "host": self.connection.host,
                    "port": self.connection.port,
                    "client_id": self.connection.client_id,
                    "market_data_type": self.connection.market_data_type,
                    "history_days": request.history_days,
                    "regime_symbols": list(request.regime_symbols),
                    "index_exchange": request.index_exchange,
                    "qualified_option_contract_count": len(option_chain.get("option_quotes", [])),
                },
            }
        finally:
            ib.disconnect()

    def _connect(self) -> Any:
        _ensure_thread_event_loop()
        try:
            from ib_insync import IB
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
        ib.reqMarketDataType(self.connection.market_data_type)
        return ib

    @staticmethod
    def _qualify_stock_contract(ib: Any, *, symbol: str, exchange: str, currency: str) -> Any:
        from ib_insync import Stock

        stock_contract = Stock(symbol, exchange, currency)
        qualified_stock = ib.qualifyContracts(stock_contract)
        if not qualified_stock:
            raise RuntimeError(f"Unable to qualify IBKR stock contract for {symbol}.")
        return qualified_stock[0]

    @staticmethod
    def _qualify_index_contract(ib: Any, *, symbol: str, exchange: str, currency: str) -> Any | None:
        from ib_insync import Index

        try:
            qualified = ib.qualifyContracts(Index(symbol, exchange, currency))
        except Exception:
            return None
        if not qualified:
            return None
        return qualified[0]

    def _request_quote_payload(self, ib: Any, contract: Any) -> tuple[dict[str, Any], float | None]:
        ticker = ib.reqTickers(contract)[0]
        last_value = _first_number(
            ticker.marketPrice(),
            ticker.last,
            ticker.close,
            ticker.bid,
            ticker.ask,
        )
        if last_value is None:
            last_value = self._request_intraday_last(ib, contract)
        close_value = _safe_float(getattr(ticker, "close", None))
        if close_value is None:
            close_value = last_value
        return (
            {
                "last": last_value,
                "close": close_value,
                "bid": _safe_float(getattr(ticker, "bid", None)),
                "ask": _safe_float(getattr(ticker, "ask", None)),
                "volume": _safe_float(getattr(ticker, "volume", None)),
            },
            last_value,
        )

    def _fetch_option_chain_snapshot(
        self,
        ib: Any,
        *,
        stock_contract: Any,
        request: IBKROptionChainRequest,
        underlying_price: float,
    ) -> dict[str, Any]:
        from ib_insync import Option

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
        option_quotes = [self._normalize_option_ticker(ticker) for ticker in tickers]
        return OptionChainSnapshot(
            underlying_symbol=request.symbol,
            underlying_price=underlying_price,
            fetched_at=_utc_now(),
            exchange=request.option_exchange,
            currency=request.currency,
            expirations=list(expirations),
            strikes=[float(strike) for strike in strikes],
            rights=list(request.rights),
            option_quotes=[quote.to_dict() for quote in option_quotes],
        ).to_dict()

    @staticmethod
    def _request_daily_history(ib: Any, contract: Any, *, history_days: int) -> list[float]:
        duration_days = max(history_days + 5, history_days)
        sec_type = str(getattr(contract, "secType", "")).upper()
        what_to_show_candidates = (
            ("TRADES", "ADJUSTED_LAST")
            if sec_type == "STK"
            else ("TRADES", "MIDPOINT")
        )
        for what_to_show in what_to_show_candidates:
            try:
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime="",
                    durationStr=f"{duration_days} D",
                    barSizeSetting="1 day",
                    whatToShow=what_to_show,
                    useRTH=False,
                    formatDate=1,
                )
            except Exception:
                continue
            closes = [
                number
                for number in (_safe_float(getattr(bar, "close", None)) for bar in bars)
                if number is not None
            ]
            if closes:
                return closes[-history_days:]
        return []

    @staticmethod
    def _request_intraday_last(ib: Any, contract: Any) -> float | None:
        sec_type = str(getattr(contract, "secType", "")).upper()
        what_to_show_candidates = (
            ("TRADES", "MIDPOINT")
            if sec_type in {"IND", "IDX"}
            else ("TRADES", "MIDPOINT")
        )
        for what_to_show in what_to_show_candidates:
            try:
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime="",
                    durationStr="1 D",
                    barSizeSetting="5 mins",
                    whatToShow=what_to_show,
                    useRTH=False,
                    formatDate=1,
                )
            except Exception:
                continue
            closes = [
                number
                for number in (_safe_float(getattr(bar, "close", None)) for bar in bars)
                if number is not None
            ]
            if closes:
                return closes[-1]
        return None

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

    def fetch_vol_regime_snapshot(
        self,
        request: IBKRVolRegimeSnapshotRequest,
    ) -> ObservationRecord:
        client = self.client_factory(self.connection)
        payload = client.fetch_vol_regime_snapshot(request)
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
