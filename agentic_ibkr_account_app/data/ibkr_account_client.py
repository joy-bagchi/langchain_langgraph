"""IBKR-backed account snapshot client."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

if sys.platform == "win32":
    from asyncio.proactor_events import _ProactorBasePipeTransport
    from functools import wraps

    _orig_call_connection_lost = _ProactorBasePipeTransport._call_connection_lost

    @wraps(_orig_call_connection_lost)
    def _patched_call_connection_lost(self, exc):
        try:
            _orig_call_connection_lost(self, exc)
        except ConnectionResetError:
            pass

    _ProactorBasePipeTransport._call_connection_lost = _patched_call_connection_lost


_IBKR_UNSET_DOUBLE = 1.7976931348623157e308
_IBKR_UNSET_INT = 2147483647


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number in {-1.0, -2.0, _IBKR_UNSET_DOUBLE, float(_IBKR_UNSET_INT)}:
        return None
    return number


def _safe_int(value: Any) -> int | None:
    number = _safe_float(value)
    if number is None:
        return None
    return int(number)


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _serialize_datetime(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    text = _safe_text(value)
    return text


def _pick_account(requested_account: str | None, managed_accounts: list[str]) -> str | None:
    normalized_requested = _safe_text(requested_account)
    if normalized_requested:
        return normalized_requested
    return managed_accounts[0] if managed_accounts else None


@dataclass(frozen=True, slots=True)
class IBKRAccountConnectionConfig:
    host: str = "127.0.0.1"
    port: int = 4001
    client_id: int = 91
    readonly: bool = True
    timeout_seconds: float = 10.0


@dataclass(frozen=True, slots=True)
class IBKRAccountSnapshotRequest:
    account: str | None = None
    include_completed_orders: bool = True
    max_fills: int = 100
    max_orders: int = 100

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "IBKRAccountSnapshotRequest":
        return cls(
            account=_safe_text(payload.get("account")) or None,
            include_completed_orders=bool(payload.get("include_completed_orders", True)),
            max_fills=max(int(payload.get("max_fills", 100)), 1),
            max_orders=max(int(payload.get("max_orders", 100)), 1),
        )


@dataclass(frozen=True, slots=True)
class IBKRAccountSnapshot:
    as_of: str
    account_id: str | None
    managed_accounts: list[str]
    dashboard: dict[str, Any]
    account_summary: list[dict[str, Any]]
    positions: list[dict[str, Any]]
    trades: list[dict[str, Any]]
    transactions: list[dict[str, Any]]
    orders: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of,
            "account_id": self.account_id,
            "managed_accounts": list(self.managed_accounts),
            "dashboard": dict(self.dashboard),
            "account_summary": list(self.account_summary),
            "positions": list(self.positions),
            "trades": list(self.trades),
            "transactions": list(self.transactions),
            "orders": list(self.orders),
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


class IBKRAccountClient(Protocol):
    def fetch_account_snapshot(self, request: IBKRAccountSnapshotRequest) -> dict[str, Any]:
        """Fetch an account-level snapshot from Interactive Brokers."""


class IBKRAccountLiveClient:
    """Real IBKR account client backed by ib_insync."""

    def __init__(self, connection: IBKRAccountConnectionConfig) -> None:
        self.connection = connection

    def fetch_account_snapshot(self, request: IBKRAccountSnapshotRequest) -> dict[str, Any]:
        from ib_insync import IB

        ib = IB()
        ib.connect(
            host=self.connection.host,
            port=self.connection.port,
            clientId=self.connection.client_id,
            readonly=self.connection.readonly,
            timeout=self.connection.timeout_seconds,
        )
        try:
            managed_accounts = list(getattr(ib, "managedAccounts", lambda: [])() or [])
            account_id = _pick_account(request.account, managed_accounts)

            summary_rows = self._normalize_account_summary(ib.accountSummary(account=account_id or ""))
            positions = self._normalize_positions(ib.positions(), account_id=account_id)
            trades = self._normalize_trades(ib.trades(), account_id=account_id, max_items=request.max_orders)
            transactions = self._normalize_transactions(ib.fills(), account_id=account_id, max_items=request.max_fills)
            orders, warnings = self._normalize_orders(
                ib=ib,
                account_id=account_id,
                include_completed_orders=request.include_completed_orders,
                max_items=request.max_orders,
            )
            dashboard = self._build_dashboard(
                account_summary=summary_rows,
                positions=positions,
                transactions=transactions,
                orders=orders,
            )
            return IBKRAccountSnapshot(
                as_of=_utc_now(),
                account_id=account_id,
                managed_accounts=managed_accounts,
                dashboard=dashboard,
                account_summary=summary_rows,
                positions=positions,
                trades=trades,
                transactions=transactions,
                orders=orders,
                warnings=warnings,
                metadata={
                    "host": self.connection.host,
                    "port": self.connection.port,
                    "client_id": self.connection.client_id,
                    "readonly": self.connection.readonly,
                },
            ).to_dict()
        finally:
            ib.disconnect()

    @staticmethod
    def _normalize_account_summary(values: list[Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in values:
            rows.append(
                {
                    "account": _safe_text(getattr(item, "account", None)),
                    "tag": _safe_text(getattr(item, "tag", None)),
                    "value": _safe_text(getattr(item, "value", None)),
                    "currency": _safe_text(getattr(item, "currency", None)),
                    "model_code": _safe_text(getattr(item, "modelCode", None)),
                }
            )
        rows.sort(key=lambda item: (item["account"], item["tag"], item["currency"]))
        return rows

    @staticmethod
    def _normalize_positions(values: list[Any], *, account_id: str | None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in values:
            account = _safe_text(getattr(item, "account", None))
            if account_id and account != account_id:
                continue
            contract = getattr(item, "contract", None)
            position = _safe_float(getattr(item, "position", None))
            market_price = _safe_float(getattr(item, "marketPrice", None))
            market_value = _safe_float(getattr(item, "marketValue", None))
            average_cost = _safe_float(getattr(item, "averageCost", None))
            unrealized_pnl = _safe_float(getattr(item, "unrealizedPNL", None))
            realized_pnl = _safe_float(getattr(item, "realizedPNL", None))
            rows.append(
                {
                    "account": account,
                    "symbol": _safe_text(getattr(contract, "symbol", None)),
                    "local_symbol": _safe_text(getattr(contract, "localSymbol", None)),
                    "sec_type": _safe_text(getattr(contract, "secType", None)),
                    "exchange": _safe_text(getattr(contract, "exchange", None)),
                    "currency": _safe_text(getattr(contract, "currency", None)),
                    "position": position,
                    "market_price": market_price,
                    "market_value": market_value,
                    "average_cost": average_cost,
                    "unrealized_pnl": unrealized_pnl,
                    "realized_pnl": realized_pnl,
                }
            )
        rows.sort(key=lambda item: (item["symbol"], item["sec_type"], item["local_symbol"]))
        return rows

    @staticmethod
    def _normalize_trades(values: list[Any], *, account_id: str | None, max_items: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for trade in values:
            order = getattr(trade, "order", None)
            contract = getattr(trade, "contract", None)
            order_status = getattr(trade, "orderStatus", None)
            account = _safe_text(getattr(order, "account", None))
            if account_id and account and account != account_id:
                continue
            rows.append(
                {
                    "account": account,
                    "symbol": _safe_text(getattr(contract, "symbol", None)),
                    "local_symbol": _safe_text(getattr(contract, "localSymbol", None)),
                    "sec_type": _safe_text(getattr(contract, "secType", None)),
                    "action": _safe_text(getattr(order, "action", None)),
                    "order_type": _safe_text(getattr(order, "orderType", None)),
                    "status": _safe_text(getattr(order_status, "status", None)),
                    "quantity": _safe_float(getattr(order, "totalQuantity", None)),
                    "filled": _safe_float(getattr(order_status, "filled", None)),
                    "remaining": _safe_float(getattr(order_status, "remaining", None)),
                    "avg_fill_price": _safe_float(getattr(order_status, "avgFillPrice", None)),
                    "lmt_price": _safe_float(getattr(order, "lmtPrice", None)),
                    "aux_price": _safe_float(getattr(order, "auxPrice", None)),
                    "order_id": _safe_int(getattr(order, "orderId", None)),
                    "perm_id": _safe_int(getattr(order, "permId", None)),
                }
            )
        return rows[:max_items]

    @staticmethod
    def _normalize_transactions(values: list[Any], *, account_id: str | None, max_items: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        sorted_values = sorted(
            list(values or []),
            key=lambda item: _serialize_datetime(getattr(getattr(item, "execution", None), "time", None)),
            reverse=True,
        )
        for fill in sorted_values:
            execution = getattr(fill, "execution", None)
            contract = getattr(fill, "contract", None)
            commission_report = getattr(fill, "commissionReport", None)
            account = _safe_text(getattr(execution, "acctNumber", None))
            if account_id and account and account != account_id:
                continue
            shares = _safe_float(getattr(execution, "shares", None))
            price = _safe_float(getattr(execution, "price", None))
            rows.append(
                {
                    "account": account,
                    "time": _serialize_datetime(getattr(execution, "time", None)),
                    "symbol": _safe_text(getattr(contract, "symbol", None)),
                    "local_symbol": _safe_text(getattr(contract, "localSymbol", None)),
                    "sec_type": _safe_text(getattr(contract, "secType", None)),
                    "side": _safe_text(getattr(execution, "side", None)),
                    "shares": shares,
                    "price": price,
                    "gross_amount": (shares * price) if shares is not None and price is not None else None,
                    "commission": _safe_float(getattr(commission_report, "commission", None)),
                    "realized_pnl": _safe_float(getattr(commission_report, "realizedPNL", None)),
                    "currency": _safe_text(getattr(commission_report, "currency", None) or getattr(contract, "currency", None)),
                    "exchange": _safe_text(getattr(execution, "exchange", None)),
                    "exec_id": _safe_text(getattr(execution, "execId", None)),
                    "order_id": _safe_int(getattr(execution, "orderId", None)),
                    "perm_id": _safe_int(getattr(execution, "permId", None)),
                }
            )
        return rows[:max_items]

    @staticmethod
    def _normalize_orders(
        *,
        ib: Any,
        account_id: str | None,
        include_completed_orders: bool,
        max_items: int,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        warnings: list[str] = []
        combined: list[dict[str, Any]] = []

        for trade in list(getattr(ib, "openTrades", lambda: [])() or []):
            order = getattr(trade, "order", None)
            account = _safe_text(getattr(order, "account", None))
            if account_id and account and account != account_id:
                continue
            contract = getattr(trade, "contract", None)
            order_status = getattr(trade, "orderStatus", None)
            combined.append(
                {
                    "source": "open_trade",
                    "account": account,
                    "symbol": _safe_text(getattr(contract, "symbol", None)),
                    "local_symbol": _safe_text(getattr(contract, "localSymbol", None)),
                    "sec_type": _safe_text(getattr(contract, "secType", None)),
                    "action": _safe_text(getattr(order, "action", None)),
                    "order_type": _safe_text(getattr(order, "orderType", None)),
                    "status": _safe_text(getattr(order_status, "status", None)),
                    "quantity": _safe_float(getattr(order, "totalQuantity", None)),
                    "filled": _safe_float(getattr(order_status, "filled", None)),
                    "remaining": _safe_float(getattr(order_status, "remaining", None)),
                    "limit_price": _safe_float(getattr(order, "lmtPrice", None)),
                    "aux_price": _safe_float(getattr(order, "auxPrice", None)),
                    "tif": _safe_text(getattr(order, "tif", None)),
                    "order_id": _safe_int(getattr(order, "orderId", None)),
                    "perm_id": _safe_int(getattr(order, "permId", None)),
                }
            )

        if include_completed_orders:
            try:
                completed = list(ib.completedOrders(apiOnly=False) or [])
            except Exception as exc:
                warnings.append(f"completedOrders unavailable: {exc}")
                completed = []
            for trade in completed:
                order = getattr(trade, "order", None)
                account = _safe_text(getattr(order, "account", None))
                if account_id and account and account != account_id:
                    continue
                contract = getattr(trade, "contract", None)
                order_status = getattr(trade, "orderStatus", None)
                combined.append(
                    {
                        "source": "completed_order",
                        "account": account,
                        "symbol": _safe_text(getattr(contract, "symbol", None)),
                        "local_symbol": _safe_text(getattr(contract, "localSymbol", None)),
                        "sec_type": _safe_text(getattr(contract, "secType", None)),
                        "action": _safe_text(getattr(order, "action", None)),
                        "order_type": _safe_text(getattr(order, "orderType", None)),
                        "status": _safe_text(getattr(order_status, "status", None)),
                        "quantity": _safe_float(getattr(order, "totalQuantity", None)),
                        "filled": _safe_float(getattr(order_status, "filled", None)),
                        "remaining": _safe_float(getattr(order_status, "remaining", None)),
                        "limit_price": _safe_float(getattr(order, "lmtPrice", None)),
                        "aux_price": _safe_float(getattr(order, "auxPrice", None)),
                        "tif": _safe_text(getattr(order, "tif", None)),
                        "order_id": _safe_int(getattr(order, "orderId", None)),
                        "perm_id": _safe_int(getattr(order, "permId", None)),
                    }
                )

        return combined[:max_items], warnings

    @staticmethod
    def _build_dashboard(
        *,
        account_summary: list[dict[str, Any]],
        positions: list[dict[str, Any]],
        transactions: list[dict[str, Any]],
        orders: list[dict[str, Any]],
    ) -> dict[str, Any]:
        summary_map: dict[str, str] = {}
        for row in account_summary:
            tag = row.get("tag")
            if tag and tag not in summary_map:
                summary_map[str(tag)] = str(row.get("value", ""))

        long_positions = sum(1 for row in positions if float(row.get("position") or 0.0) > 0)
        short_positions = sum(1 for row in positions if float(row.get("position") or 0.0) < 0)
        total_market_value = sum(float(row.get("market_value") or 0.0) for row in positions)
        unrealized_pnl = sum(float(row.get("unrealized_pnl") or 0.0) for row in positions)
        realized_pnl = sum(float(row.get("realized_pnl") or 0.0) for row in positions)

        recent_transaction_count = len(transactions)
        buy_transaction_count = sum(1 for row in transactions if str(row.get("side", "")).upper() == "BOT")
        sell_transaction_count = sum(1 for row in transactions if str(row.get("side", "")).upper() == "SLD")
        open_order_count = sum(1 for row in orders if str(row.get("source")) == "open_trade")

        return {
            "net_liquidation": summary_map.get("NetLiquidation"),
            "available_funds": summary_map.get("AvailableFunds"),
            "buying_power": summary_map.get("BuyingPower"),
            "equity_with_loan_value": summary_map.get("EquityWithLoanValue"),
            "gross_position_value": summary_map.get("GrossPositionValue"),
            "excess_liquidity": summary_map.get("ExcessLiquidity"),
            "init_margin_req": summary_map.get("InitMarginReq"),
            "maint_margin_req": summary_map.get("MaintMarginReq"),
            "position_count": len(positions),
            "long_position_count": long_positions,
            "short_position_count": short_positions,
            "total_market_value": total_market_value,
            "positions_unrealized_pnl": unrealized_pnl,
            "positions_realized_pnl": realized_pnl,
            "recent_transaction_count": recent_transaction_count,
            "buy_transaction_count": buy_transaction_count,
            "sell_transaction_count": sell_transaction_count,
            "open_order_count": open_order_count,
        }


@dataclass(slots=True)
class IBKRAccountDataPipe:
    """Stable high-level data pipe over the lower-level IBKR account client."""

    connection: IBKRAccountConnectionConfig = field(default_factory=IBKRAccountConnectionConfig)
    client_factory: type[IBKRAccountClient] | Any = IBKRAccountLiveClient

    def fetch_account_snapshot(self, request: IBKRAccountSnapshotRequest) -> IBKRAccountSnapshot:
        client = self.client_factory(self.connection)
        payload = client.fetch_account_snapshot(request)
        return IBKRAccountSnapshot(
            as_of=str(payload.get("as_of", _utc_now())),
            account_id=payload.get("account_id"),
            managed_accounts=list(payload.get("managed_accounts", [])),
            dashboard=dict(payload.get("dashboard", {})),
            account_summary=list(payload.get("account_summary", [])),
            positions=list(payload.get("positions", [])),
            trades=list(payload.get("trades", [])),
            transactions=list(payload.get("transactions", [])),
            orders=list(payload.get("orders", [])),
            warnings=list(payload.get("warnings", [])),
            metadata=dict(payload.get("metadata", {})),
        )
