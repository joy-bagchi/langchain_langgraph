"""IBKR-backed market data pipe for underlying quotes and option chains."""

from __future__ import annotations

import asyncio
import math
import sys
import time as _time
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Callable, Protocol

# Patch for Windows asyncio ConnectionResetError: [WinError 10054]
if sys.platform == "win32":
    from functools import wraps
    from asyncio.proactor_events import _ProactorBasePipeTransport

    _orig_call_connection_lost = _ProactorBasePipeTransport._call_connection_lost

    @wraps(_orig_call_connection_lost)
    def _patched_call_connection_lost(self, exc):
        try:
            _orig_call_connection_lost(self, exc)
        except ConnectionResetError:
            # On Windows, this error is common when the remote host (TWS/Gateway)
            # forcibly closes the connection. It can be safely ignored during
            # the connection loss sequence.
            pass

    _ProactorBasePipeTransport._call_connection_lost = _patched_call_connection_lost


from agentic_vol_regime_app.contracts import (
    ObservationRecord,
    OptionChainSnapshot,
    OptionGreekRecord,
    OptionQuoteRecord,
)


DEFAULT_VOL_REGIME_SYMBOLS = ("SPY", "VIX", "VVIX", "VIX9D", "VIX3M", "VIX6M")
DEFAULT_SECTOR_ETF_SYMBOLS = ("XLK", "XLF", "XLE", "XLY", "XLP", "XLI", "XLB", "XLV", "XLU", "XLRE")
INDEX_STYLE_SYMBOLS = {"VIX", "VVIX", "VIX9D", "VIX3M", "VIX6M", "VIX9M"}
_IBKR_UNSET_DOUBLE = 1.7976931348623157e308
_IBKR_UNSET_INT = 2147483647
_CHUNKED_HISTORY_THRESHOLD_DAYS = 1260


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


def _historical_duration_str(history_days: int) -> str:
    normalized_days = max(int(history_days), 0)
    if normalized_days <= 365:
        duration_days = max(normalized_days + 5, normalized_days)
        return f"{duration_days} D"
    duration_years = max(1, math.ceil(normalized_days / 252))
    return f"{duration_years} Y"


def _historical_duration_str_for_calendar_days(calendar_days: int) -> str:
    normalized_days = max(int(calendar_days), 1)
    if normalized_days <= 365:
        return f"{normalized_days} D"
    duration_years = max(1, math.ceil(normalized_days / 365))
    return f"{duration_years} Y"


def _parse_as_of_date(value: Any) -> date:
    text = str(value).strip()
    if not text:
        raise ValueError("Expected a non-empty as_of_date.")
    try:
        parsed = date.fromisoformat(text[:10])
    except ValueError as exc:
        raise ValueError("Expected as_of_date in YYYY-MM-DD format.") from exc
    while parsed.weekday() >= 5:
        parsed -= timedelta(days=1)
    return parsed


def _historical_end_datetime(as_of_date: str) -> str:
    normalized_day = _parse_as_of_date(as_of_date)
    end_dt = datetime.combine(normalized_day, time(hour=23, minute=59, second=59), tzinfo=timezone.utc)
    return end_dt.strftime("%Y%m%d %H:%M:%S UTC")


def _historical_as_of_timestamp(as_of_date: str) -> str:
    normalized_day = _parse_as_of_date(as_of_date)
    as_of_dt = datetime.combine(normalized_day, time(hour=21, minute=0, second=0), tzinfo=timezone.utc)
    return as_of_dt.isoformat().replace("+00:00", "Z")


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
    history_days: int = 252
    regime_symbols: tuple[str, ...] = DEFAULT_VOL_REGIME_SYMBOLS
    index_exchange: str = "CBOE"
    currency: str = "USD"
    as_of_date: str | None = None

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
            history_days=max(int(payload.get("history_days", 252)), 0),
            regime_symbols=normalized_symbols or DEFAULT_VOL_REGIME_SYMBOLS,
            index_exchange=str(payload.get("index_exchange", "CBOE")),
            currency=str(payload.get("currency", option_chain.currency or "USD")),
            as_of_date=(str(payload.get("as_of_date")).strip() or None) if payload.get("as_of_date") else None,
        )


@dataclass(frozen=True, slots=True)
class IBKRDailyBar:
    symbol: str
    session_date: date
    close: float
    actual_what_to_show: str
    source: str = "IBKR"


@dataclass(frozen=True, slots=True)
class IBKRDailyHistoryRequest:
    symbol: str
    end_date: date | None = None
    duration_calendar_days: int = 30
    exchange: str = "SMART"
    currency: str = "USD"
    use_rth: bool = True
    preferred_what_to_show: str = "ADJUSTED_LAST"
    allow_fallback_to_trades: bool = True


@dataclass(frozen=True, slots=True)
class IBKRDailyHistoryResult:
    bars: tuple[IBKRDailyBar, ...]
    actual_what_to_show: str
    warnings: tuple[str, ...] = ()


class IBKRMarketDataClient(Protocol):
    """Low-level market data client contract."""

    def fetch_market_snapshot(self, request: IBKROptionChainRequest) -> dict[str, Any]:
        """Fetch an underlying quote and selected option-chain data."""

    def fetch_vol_regime_snapshot(self, request: IBKRVolRegimeSnapshotRequest) -> dict[str, Any]:
        """Fetch the full daily volatility-regime observation payload."""

    def request_daily_bars(self, request: IBKRDailyHistoryRequest) -> IBKRDailyHistoryResult:
        """Fetch dated daily bars for a symbol."""


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
            if request.as_of_date:
                return self._fetch_historical_vol_regime_snapshot(ib, request=request)

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
                is_sector_etf = normalized_symbol in DEFAULT_SECTOR_ETF_SYMBOLS
                if normalized_symbol == option_request.symbol.upper():
                    contract = stock_contract
                    quote_payload = underlying_quote
                else:
                    if normalized_symbol in INDEX_STYLE_SYMBOLS:
                        contract = self._qualify_index_contract(
                            ib,
                            symbol=normalized_symbol,
                            exchange=request.index_exchange,
                            currency=request.currency,
                        )
                    else:
                        try:
                            contract = self._qualify_stock_contract(
                                ib,
                                symbol=normalized_symbol,
                                exchange="SMART" if is_sector_etf else option_request.exchange,
                                currency="USD" if is_sector_etf else request.currency,
                            )
                        except Exception:
                            contract = None
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
                    if normalized_symbol == option_request.symbol.upper() or normalized_symbol not in INDEX_STYLE_SYMBOLS
                    else normalized_symbol
                )
                if request.history_days > 0:
                    use_rth_for_history = is_sector_etf or (normalized_symbol == option_request.symbol.upper())
                    history_values, history_warnings = self._request_daily_history(
                        ib,
                        contract,
                        history_days=request.history_days,
                        history_label=history_key,
                        use_rth=use_rth_for_history,
                        force_chunked=is_sector_etf and request.history_days > 252,
                    )
                    if history_values:
                        history_payload[history_key] = history_values
                    else:
                        warnings.extend(history_warnings or [f"unable to load daily history for {history_key}"])

            missing_symbols = [
                symbol for symbol in request.regime_symbols if symbol.upper() not in symbols_payload
            ]
            required_history = [
                ("SPY_close" if symbol.upper() == option_request.symbol.upper() else symbol.upper())
                for symbol in request.regime_symbols
            ]
            missing_history: list[str] = []
            if request.history_days > 0:
                minimum_history = min(request.history_days, 22)
                missing_history = [
                    key for key in required_history if len(history_payload.get(key, [])) < minimum_history
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
                    "history_fetch_mode": "skipped" if request.history_days <= 0 else "full_or_incremental",
                    "regime_symbols": list(request.regime_symbols),
                    "index_exchange": request.index_exchange,
                    "qualified_option_contract_count": len(option_chain.get("option_quotes", [])),
                },
            }
        finally:
            ib.disconnect()

    def request_daily_bars(self, request: IBKRDailyHistoryRequest) -> IBKRDailyHistoryResult:
        ib = self._connect()
        try:
            contract = self._qualify_stock_contract(
                ib,
                symbol=request.symbol,
                exchange=request.exchange,
                currency=request.currency,
            )
            end_datetime = ""
            if request.end_date is not None:
                end_dt = datetime.combine(request.end_date, time(hour=23, minute=59, second=59), tzinfo=timezone.utc)
                end_datetime = end_dt.strftime("%Y%m%d %H:%M:%S UTC")
            bars, actual_what_to_show, warnings = self._request_daily_history_bars(
                ib,
                contract,
                symbol=str(request.symbol).upper(),
                duration_calendar_days=max(int(request.duration_calendar_days), 1),
                history_label=f"{str(request.symbol).upper()}_close",
                end_datetime=end_datetime,
                use_rth=bool(request.use_rth),
                preferred_what_to_show=str(request.preferred_what_to_show).upper() or "ADJUSTED_LAST",
                allow_fallback_to_trades=bool(request.allow_fallback_to_trades),
            )
            if not bars:
                raise RuntimeError(
                    "IBKR returned no usable dated daily bars for "
                    f"{str(request.symbol).upper()}. {' | '.join(warnings) if warnings else ''}".strip()
                )
            return IBKRDailyHistoryResult(
                bars=tuple(bars),
                actual_what_to_show=actual_what_to_show,
                warnings=tuple(warnings),
            )
        finally:
            ib.disconnect()

    def _fetch_historical_vol_regime_snapshot(
        self,
        ib: Any,
        *,
        request: IBKRVolRegimeSnapshotRequest,
    ) -> dict[str, Any]:
        option_request = request.option_chain
        requested_as_of = str(request.as_of_date or "").strip()
        as_of_timestamp = _historical_as_of_timestamp(requested_as_of)
        end_datetime = _historical_end_datetime(requested_as_of)

        stock_contract = self._qualify_stock_contract(
            ib,
            symbol=option_request.symbol,
            exchange=option_request.exchange,
            currency=option_request.currency,
        )

        symbols_payload: dict[str, dict[str, Any]] = {}
        history_payload: dict[str, list[float]] = {}
        warnings: list[str] = []
        missing_symbols: list[str] = []

        for symbol in request.regime_symbols:
            normalized_symbol = symbol.upper()
            is_sector_etf = normalized_symbol in DEFAULT_SECTOR_ETF_SYMBOLS
            if normalized_symbol == option_request.symbol.upper():
                contract = stock_contract
            elif normalized_symbol in INDEX_STYLE_SYMBOLS:
                contract = self._qualify_index_contract(
                    ib,
                    symbol=normalized_symbol,
                    exchange=request.index_exchange,
                    currency=request.currency,
                )
            else:
                try:
                    contract = self._qualify_stock_contract(
                        ib,
                        symbol=normalized_symbol,
                        exchange="SMART" if is_sector_etf else option_request.exchange,
                        currency="USD" if is_sector_etf else request.currency,
                    )
                except Exception:
                    contract = None

            if contract is None:
                missing_symbols.append(normalized_symbol)
                warnings.append(f"unable to qualify contract for {normalized_symbol}")
                continue

            history_key = (
                f"{normalized_symbol}_close"
                if normalized_symbol == option_request.symbol.upper() or normalized_symbol not in INDEX_STYLE_SYMBOLS
                else normalized_symbol
            )
            history_values, history_warnings = self._request_daily_history(
                ib,
                contract,
                history_days=request.history_days,
                history_label=history_key,
                end_datetime=end_datetime,
                use_rth=(is_sector_etf or normalized_symbol == option_request.symbol.upper()),
                force_chunked=is_sector_etf and request.history_days > 252,
            )
            if history_values:
                history_payload[history_key] = history_values
                close_value = float(history_values[-1])
                symbols_payload[normalized_symbol] = {
                    "last": close_value,
                    "close": close_value,
                    "bid": None,
                    "ask": None,
                    "volume": None,
                }
            else:
                warnings.extend(history_warnings or [f"unable to load daily history for {history_key}"])

        underlying_symbol = option_request.symbol.upper()
        underlying_history_key = f"{underlying_symbol}_close"
        underlying_price = None
        if history_payload.get(underlying_history_key):
            underlying_price = float(history_payload[underlying_history_key][-1])

        required_history = [
            ("SPY_close" if symbol.upper() == underlying_symbol else symbol.upper())
            for symbol in request.regime_symbols
        ]
        minimum_history = min(request.history_days, 22) if request.history_days > 0 else 0
        missing_history = [
            key for key in required_history if len(history_payload.get(key, [])) < minimum_history
        ]

        quality = {
            "is_complete": not missing_symbols and not missing_history,
            "warnings": warnings,
            "stale_fields": [],
            "missing_symbols": missing_symbols,
            "missing_history": missing_history,
        }

        option_chain = {
            "underlying_symbol": option_request.symbol,
            "underlying_price": underlying_price,
            "fetched_at": as_of_timestamp,
            "exchange": option_request.option_exchange,
            "currency": option_request.currency,
            "expirations": [],
            "strikes": [],
            "rights": list(option_request.rights),
            "option_quotes": [],
        }

        return {
            "schema_version": "observation.v1",
            "as_of": as_of_timestamp,
            "source": "IBKR_HISTORICAL_AS_OF",
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
                "history_fetch_mode": "historical_as_of",
                "regime_symbols": list(request.regime_symbols),
                "index_exchange": request.index_exchange,
                "requested_as_of_date": requested_as_of,
            },
        }

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
        selected_exchange = str(getattr(selected_parameters, "exchange", "") or request.option_exchange or "SMART")
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
                            selected_exchange,
                            multiplier=str(getattr(selected_parameters, "multiplier", "") or ""),
                            currency=request.currency,
                            tradingClass=getattr(selected_parameters, "tradingClass", None),
                        )
                    )
        if not option_contracts:
            raise RuntimeError("No option contracts were selected from the IBKR chain parameters.")

        qualified_option_contracts: list[Any] = []
        qualification_warnings: list[str] = []
        for option_contract in option_contracts:
            expiry = str(getattr(option_contract, "lastTradeDateOrContractMonth", ""))
            strike = getattr(option_contract, "strike", None)
            right = str(getattr(option_contract, "right", ""))
            try:
                qualified = ib.qualifyContracts(option_contract)
            except Exception as exc:  # pragma: no cover - depends on broker response
                qualification_warnings.append(
                    f"Skipping invalid option contract {request.symbol} {expiry} {strike} {right}: {exc}"
                )
                continue
            if not qualified:
                qualification_warnings.append(
                    f"Skipping unqualified option contract {request.symbol} {expiry} {strike} {right}: IBKR returned no contract details."
                )
                continue
            qualified_option_contracts.append(qualified[0])

        if not qualified_option_contracts:
            warning_text = "; ".join(qualification_warnings) if qualification_warnings else "IBKR rejected all generated option contracts."
            raise RuntimeError(f"No valid option contracts were qualified for {request.symbol}. {warning_text}")

        tickers = list(ib.reqTickers(*qualified_option_contracts))
        option_quotes = [self._normalize_option_ticker(ticker) for ticker in tickers]
        return OptionChainSnapshot(
            underlying_symbol=request.symbol,
            underlying_price=underlying_price,
            fetched_at=_utc_now(),
            exchange=selected_exchange,
            currency=request.currency,
            expirations=list(expirations),
            strikes=[float(strike) for strike in strikes],
            rights=list(request.rights),
            option_quotes=[quote.to_dict() for quote in option_quotes],
            warnings=qualification_warnings,
        ).to_dict()

    @staticmethod
    def _request_daily_history(
        ib: Any,
        contract: Any,
        *,
        history_days: int,
        history_label: str,
        end_datetime: str = "",
        use_rth: bool = False,
        force_chunked: bool = False,
    ) -> tuple[list[float], list[str]]:
        dated_bars, _actual_what_to_show, diagnostics = IBKRLiveClient._request_daily_history_bars(
            ib,
            contract,
            symbol=str(getattr(contract, "symbol", history_label)).replace("_close", "").upper(),
            duration_calendar_days=max(int(history_days) + 5, 1),
            history_label=history_label,
            end_datetime=end_datetime,
            use_rth=use_rth,
            preferred_what_to_show="ADJUSTED_LAST",
            allow_fallback_to_trades=True,
            force_chunked=force_chunked,
            target_trading_days=history_days,
        )
        if not dated_bars:
            return [], diagnostics
        closes = [float(bar.close) for bar in dated_bars]
        return closes[-history_days:], diagnostics

    @staticmethod
    def _history_what_to_show_candidates(
        *,
        sec_type: str,
        preferred_what_to_show: str,
        allow_fallback_to_trades: bool,
        end_datetime: str,
    ) -> tuple[str, ...]:
        normalized_sec_type = str(sec_type).upper()
        normalized_preference = str(preferred_what_to_show).upper()
        if normalized_sec_type == "STK":
            if normalized_preference == "ADJUSTED_LAST" and str(end_datetime).strip():
                ordered = ["TRADES"]
                if allow_fallback_to_trades:
                    ordered.append("ADJUSTED_LAST")
                return tuple(dict.fromkeys(item for item in ordered if item))
            ordered = [normalized_preference or "ADJUSTED_LAST"]
            if allow_fallback_to_trades and "TRADES" not in ordered:
                ordered.append("TRADES")
            return tuple(dict.fromkeys(item for item in ordered if item))
        if normalized_sec_type in {"IND", "IDX"}:
            return ("TRADES", "MIDPOINT")
        return ("TRADES",)

    @staticmethod
    def _normalize_bar_date(value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
        text = str(value).strip()
        if not text:
            return None
        if " " in text and "-" in text:
            text = text.split(" ")[0]
        for fmt in ("%Y%m%d", "%Y-%m-%d"):
            try:
                parsed = datetime.strptime(text, fmt)
                return parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    @staticmethod
    def _bars_to_daily_bars(
        bars: list[Any],
        *,
        symbol: str,
        actual_what_to_show: str,
    ) -> list[IBKRDailyBar]:
        by_date: dict[date, IBKRDailyBar] = {}
        for bar in bars:
            close_value = _safe_float(getattr(bar, "close", None))
            bar_dt = IBKRLiveClient._normalize_bar_date(getattr(bar, "date", None))
            if close_value is None or bar_dt is None:
                continue
            by_date[bar_dt.date()] = IBKRDailyBar(
                symbol=str(symbol).upper(),
                session_date=bar_dt.date(),
                close=float(close_value),
                actual_what_to_show=str(actual_what_to_show).upper(),
                source="IBKR",
            )
        return [by_date[item] for item in sorted(by_date.keys())]

    @staticmethod
    def _bars_to_sorted_closes(bars: list[Any]) -> list[float]:
        daily_bars = IBKRLiveClient._bars_to_daily_bars(
            bars,
            symbol=str(getattr(bars[0], "symbol", "UNKNOWN")) if bars else "UNKNOWN",
            actual_what_to_show="TRADES",
        )
        return [float(bar.close) for bar in daily_bars]

    @staticmethod
    def _request_daily_history_bars(
        ib: Any,
        contract: Any,
        *,
        symbol: str,
        duration_calendar_days: int,
        history_label: str,
        end_datetime: str = "",
        use_rth: bool = False,
        preferred_what_to_show: str = "ADJUSTED_LAST",
        allow_fallback_to_trades: bool = True,
        force_chunked: bool = False,
        target_trading_days: int | None = None,
    ) -> tuple[list[IBKRDailyBar], str, list[str]]:
        duration_str = _historical_duration_str_for_calendar_days(duration_calendar_days)
        sec_type = str(getattr(contract, "secType", "")).upper()
        should_use_chunked = bool(force_chunked) or (
            sec_type == "STK" and int(target_trading_days or 0) > _CHUNKED_HISTORY_THRESHOLD_DAYS
        )
        diagnostics: list[str] = []
        actual_what_to_show = ""
        what_to_show_candidates = IBKRLiveClient._history_what_to_show_candidates(
            sec_type=sec_type,
            preferred_what_to_show=preferred_what_to_show,
            allow_fallback_to_trades=allow_fallback_to_trades,
            end_datetime=end_datetime,
        )
        for what_to_show in what_to_show_candidates:
            actual_what_to_show = what_to_show
            if should_use_chunked and sec_type == "STK":
                stitched, chunk_warnings = IBKRLiveClient._request_daily_history_chunked_bars(
                    ib,
                    contract,
                    symbol=symbol,
                    history_label=history_label,
                    what_to_show=what_to_show,
                    end_datetime=end_datetime,
                    use_rth=bool(use_rth),
                    target_trading_days=target_trading_days,
                )
                if stitched:
                    return stitched, what_to_show, chunk_warnings
            try:
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime=end_datetime,
                    durationStr=duration_str,
                    barSizeSetting="1 day",
                    whatToShow=what_to_show,
                    useRTH=bool(use_rth),
                    formatDate=1,
                )
            except Exception as exc:
                diagnostics.append(
                    f"{history_label} history request failed for {what_to_show} with duration {duration_str}: {exc}"
                )
                error_text = str(exc).lower()
                if sec_type == "STK" and any(token in error_text for token in ("timeout", "cancelled", "canceled")):
                    stitched, chunk_warnings = IBKRLiveClient._request_daily_history_chunked_bars(
                        ib,
                        contract,
                        symbol=symbol,
                        history_label=history_label,
                        what_to_show=what_to_show,
                        end_datetime=end_datetime,
                        use_rth=bool(use_rth),
                        target_trading_days=target_trading_days,
                    )
                    if stitched:
                        diagnostics.extend(chunk_warnings)
                        return stitched, what_to_show, diagnostics
                continue
            daily_bars = IBKRLiveClient._bars_to_daily_bars(
                bars,
                symbol=symbol,
                actual_what_to_show=what_to_show,
            )
            if target_trading_days is not None and len(daily_bars) >= int(target_trading_days):
                return daily_bars[-int(target_trading_days):], what_to_show, diagnostics
            if daily_bars and not force_chunked:
                return daily_bars, what_to_show, diagnostics
            if sec_type == "STK":
                stitched, chunk_warnings = IBKRLiveClient._request_daily_history_chunked_bars(
                    ib,
                    contract,
                    symbol=symbol,
                    history_label=history_label,
                    what_to_show=what_to_show,
                    end_datetime=end_datetime,
                    use_rth=bool(use_rth),
                    target_trading_days=target_trading_days,
                )
                if stitched:
                    diagnostics.extend(chunk_warnings)
                    return stitched, what_to_show, diagnostics
            diagnostics.append(
                f"{history_label} history request returned no usable bars for {what_to_show} with duration {duration_str}."
            )
        return [], actual_what_to_show or preferred_what_to_show, diagnostics

    @staticmethod
    def _request_daily_history_chunked(
        ib: Any,
        contract: Any,
        *,
        history_days: int,
        history_label: str,
        what_to_show: str,
        end_datetime: str,
        use_rth: bool,
    ) -> tuple[list[float], list[str]]:
        bars, warnings = IBKRLiveClient._request_daily_history_chunked_bars(
            ib,
            contract,
            symbol=str(getattr(contract, "symbol", history_label)).replace("_close", "").upper(),
            history_label=history_label,
            what_to_show=what_to_show,
            end_datetime=end_datetime,
            use_rth=use_rth,
            target_trading_days=history_days,
        )
        return [float(bar.close) for bar in bars], warnings

    @staticmethod
    def _request_daily_history_chunked_bars(
        ib: Any,
        contract: Any,
        *,
        symbol: str,
        history_label: str,
        what_to_show: str,
        end_datetime: str,
        use_rth: bool,
        target_trading_days: int | None,
    ) -> tuple[list[IBKRDailyBar], list[str]]:
        if max(int(target_trading_days or 0), 0) <= 0:
            return [], []
        diagnostics: list[str] = []
        stitched_by_date: dict[date, IBKRDailyBar] = {}
        if end_datetime:
            current_end = end_datetime
        else:
            current_end = datetime.now(timezone.utc).strftime("%Y%m%d %H:%M:%S UTC")
        max_chunks = 24
        duration_candidates = ("1 Y", "6 M", "3 M")
        retries_per_duration = 2
        for _ in range(max_chunks):
            bars: list[Any] = []
            for duration_str in duration_candidates:
                attempt = 0
                while attempt <= retries_per_duration:
                    try:
                        bars = ib.reqHistoricalData(
                            contract,
                            endDateTime=current_end,
                            durationStr=duration_str,
                            barSizeSetting="1 day",
                            whatToShow=what_to_show,
                            useRTH=bool(use_rth),
                            formatDate=1,
                        )
                        if bars:
                            break
                    except Exception as exc:
                        diagnostics.append(
                            f"{history_label} chunked history request failed for {what_to_show} "
                            f"(duration={duration_str}, endDateTime={current_end}, attempt={attempt + 1}): {exc}"
                        )
                        _time.sleep(0.35 * (attempt + 1))
                    attempt += 1
                if bars:
                    break
            if not bars:
                break
            rows = IBKRLiveClient._bars_to_daily_bars(
                bars,
                symbol=symbol,
                actual_what_to_show=what_to_show,
            )
            if not rows:
                break
            for row in rows:
                stitched_by_date[row.session_date] = row
            if len(stitched_by_date) >= int(target_trading_days or 0):
                break
            earliest = min(row.session_date for row in rows)
            next_end_dt = datetime.combine(earliest - timedelta(days=1), time(23, 59, 59), tzinfo=timezone.utc)
            current_end = next_end_dt.strftime("%Y%m%d %H:%M:%S UTC")
            _time.sleep(0.15)
        if not stitched_by_date:
            return [], diagnostics
        ordered_days = sorted(stitched_by_date.keys())
        daily_bars = [stitched_by_date[item] for item in ordered_days]
        if len(daily_bars) < int(target_trading_days or 0):
            diagnostics.append(
                f"{history_label} chunked history returned {len(daily_bars)} rows, below requested {int(target_trading_days or 0)} rows."
            )
        return daily_bars[-int(target_trading_days or 0):], diagnostics

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
            if not selected:
                selected = available_sorted[:limit]
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
        ranked = sorted(
            sorted_strikes,
            key=lambda strike: (
                1 if abs(strike - round(strike)) > 1e-9 else 0,
                abs(strike - underlying_price),
                strike,
            ),
        )
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

    def request_daily_bars(
        self,
        request: IBKRDailyHistoryRequest,
    ) -> IBKRDailyHistoryResult:
        client = self.client_factory(self.connection)
        return client.request_daily_bars(request)
