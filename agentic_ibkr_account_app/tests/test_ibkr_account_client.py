from __future__ import annotations

from datetime import datetime, timezone

from agentic_ibkr_account_app.data.ibkr_account_client import (
    IBKRAccountConnectionConfig,
    IBKRAccountDataPipe,
    IBKRAccountSnapshotRequest,
)


class _FakeSummaryRow:
    def __init__(self, account: str, tag: str, value: str, currency: str = "USD") -> None:
        self.account = account
        self.tag = tag
        self.value = value
        self.currency = currency
        self.modelCode = ""


class _FakeContract:
    def __init__(self, symbol: str, sec_type: str = "STK", local_symbol: str | None = None) -> None:
        self.symbol = symbol
        self.localSymbol = local_symbol or symbol
        self.secType = sec_type
        self.exchange = "SMART"
        self.currency = "USD"


class _FakePosition:
    def __init__(self, account: str, symbol: str, position: float, market_value: float) -> None:
        self.account = account
        self.contract = _FakeContract(symbol)
        self.position = position
        self.marketPrice = market_value / position
        self.marketValue = market_value
        self.averageCost = self.marketPrice - 10.0
        self.unrealizedPNL = 25.0
        self.realizedPNL = 5.0


class _FakeOrder:
    def __init__(self, account: str, action: str, quantity: float, order_id: int) -> None:
        self.account = account
        self.action = action
        self.orderType = "LMT"
        self.totalQuantity = quantity
        self.lmtPrice = 123.45
        self.auxPrice = None
        self.tif = "DAY"
        self.orderId = order_id
        self.permId = order_id + 1000


class _FakeOrderStatus:
    def __init__(self, status: str, filled: float, remaining: float, avg_fill_price: float) -> None:
        self.status = status
        self.filled = filled
        self.remaining = remaining
        self.avgFillPrice = avg_fill_price


class _FakeTrade:
    def __init__(self, account: str, symbol: str, action: str, status: str, order_id: int) -> None:
        self.order = _FakeOrder(account, action, 10, order_id)
        self.orderStatus = _FakeOrderStatus(status, 5, 5, 120.0)
        self.contract = _FakeContract(symbol)


class _FakeExecution:
    def __init__(self, account: str, side: str, shares: float, price: float, order_id: int) -> None:
        self.acctNumber = account
        self.time = datetime(2026, 6, 22, 12, 0, tzinfo=timezone.utc)
        self.side = side
        self.shares = shares
        self.price = price
        self.exchange = "NYSE"
        self.execId = f"exec-{order_id}"
        self.orderId = order_id
        self.permId = order_id + 1000


class _FakeCommissionReport:
    def __init__(self) -> None:
        self.commission = 1.25
        self.realizedPNL = 12.5
        self.currency = "USD"


class _FakeFill:
    def __init__(self, account: str, symbol: str, side: str, shares: float, price: float, order_id: int) -> None:
        self.execution = _FakeExecution(account, side, shares, price, order_id)
        self.contract = _FakeContract(symbol)
        self.commissionReport = _FakeCommissionReport()


class _FakeClient:
    def __init__(self, _connection: IBKRAccountConnectionConfig) -> None:
        pass

    def fetch_account_snapshot(self, _request: IBKRAccountSnapshotRequest) -> dict:
        return {
            "as_of": "2026-06-22T12:00:00+00:00",
            "account_id": "DU123",
            "managed_accounts": ["DU123"],
            "dashboard": {
                "net_liquidation": "100000",
                "position_count": 2,
                "open_order_count": 1,
            },
            "account_summary": [
                _FakeSummaryRow("DU123", "NetLiquidation", "100000").__dict__,
            ],
            "positions": [
                {
                    "account": "DU123",
                    "symbol": "AAPL",
                    "position": 10.0,
                    "market_value": 2000.0,
                }
            ],
            "trades": [
                {
                    "account": "DU123",
                    "symbol": "AAPL",
                    "status": "Submitted",
                }
            ],
            "transactions": [
                {
                    "account": "DU123",
                    "symbol": "AAPL",
                    "side": "BOT",
                    "gross_amount": 1000.0,
                }
            ],
            "orders": [
                {
                    "account": "DU123",
                    "symbol": "AAPL",
                    "source": "open_trade",
                }
            ],
            "warnings": [],
            "metadata": {"port": 4001},
        }


def test_account_data_pipe_round_trip() -> None:
    pipe = IBKRAccountDataPipe(
        connection=IBKRAccountConnectionConfig(),
        client_factory=_FakeClient,
    )
    snapshot = pipe.fetch_account_snapshot(IBKRAccountSnapshotRequest())
    assert snapshot.account_id == "DU123"
    assert snapshot.dashboard["position_count"] == 2
    assert snapshot.transactions[0]["side"] == "BOT"
