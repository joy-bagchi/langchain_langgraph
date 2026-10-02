"""Read-only, model-bounded IBKR data tool contracts.

Provider-specific MCP names and schemas intentionally do not live here. The
authenticated catalog must be inspected and reviewed before a live provider is
configured; tests use the provider protocol below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from typing import Any, Protocol


READ_ONLY_SCOPE = "mcp.read"
IBKR_ACTIONS = (
    "get_symbol_daily_data",
    "list_option_contracts",
    "get_option_data",
)


class IBKRDataReaderError(RuntimeError):
    """Safe base error for the bounded IBKR reader."""


class IBKRAuthorizationRequired(IBKRDataReaderError):
    """The stored read grant needs interactive user consent."""


class IBKRScopeRejected(IBKRDataReaderError):
    """The stored or refreshed grant is not exactly read-only."""


class IBKRStorageUnavailable(IBKRDataReaderError):
    """Protected credential storage could not be read or updated."""


@dataclass(frozen=True, slots=True, repr=False)
class IBKRCredentials:
    """Opaque OAuth models. repr is disabled to keep tokens out of logs."""

    client_info: dict[str, Any] = field(repr=False)
    tokens: dict[str, Any] = field(repr=False)

    @property
    def scope(self) -> str | None:
        scope = self.tokens.get("scope")
        return str(scope) if scope is not None else None


class IBKRCredentialStore(Protocol):
    def load(self) -> IBKRCredentials | None: ...

    def persist(self, credentials: IBKRCredentials) -> None: ...


class IBKRDataProvider(Protocol):
    """Trusted adapter interface. Credentials never enter tool arguments."""

    def refresh_credentials(self, credentials: IBKRCredentials) -> IBKRCredentials: ...

    def invoke(
        self,
        action: str,
        arguments: dict[str, Any],
        credentials: IBKRCredentials,
    ) -> "ProviderResult": ...


@dataclass(frozen=True, slots=True, repr=False)
class ProviderResult:
    payload: dict[str, Any] = field(repr=False)
    rotated_credentials: IBKRCredentials | None = field(default=None, repr=False)


class SecretManagerIBKRCredentialStore:
    """Versioned Secret Manager storage using the Cloud Run runtime identity."""

    def __init__(
        self,
        *,
        project_id: str | None = None,
        client_secret_id: str | None = None,
        token_secret_id: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
        self.client_secret_id = client_secret_id or os.environ.get(
            "IBKR_OAUTH_CLIENT_SECRET_ID", ""
        ).strip()
        self.token_secret_id = token_secret_id or os.environ.get(
            "IBKR_OAUTH_TOKEN_SECRET_ID", ""
        ).strip()
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            if not (self.project_id and self.client_secret_id and self.token_secret_id):
                raise IBKRStorageUnavailable("IBKR protected credential storage is not configured")
            try:
                from google.cloud import secretmanager

                self._client = secretmanager.SecretManagerServiceClient()
            except Exception as exc:
                raise IBKRStorageUnavailable(
                    "IBKR protected credential storage is unavailable"
                ) from exc
        return self._client

    def _secret_name(self, secret_id: str) -> str:
        if not (self.project_id and secret_id):
            raise IBKRStorageUnavailable("IBKR protected credential storage is not configured")
        return f"projects/{self.project_id}/secrets/{secret_id}"

    def _read_json(self, secret_id: str) -> dict[str, Any] | None:
        client = self._get_client()
        name = self._secret_name(secret_id) + "/versions/latest"
        try:
            response = client.access_secret_version(request={"name": name})
        except Exception as exc:
            if type(exc).__name__ == "NotFound":
                return None
            raise IBKRStorageUnavailable("IBKR protected credential storage is unavailable") from exc
        try:
            value = json.loads(response.payload.data.decode("utf-8"))
        except Exception as exc:
            raise IBKRStorageUnavailable("Stored IBKR credential data is invalid") from exc
        if not isinstance(value, dict):
            raise IBKRStorageUnavailable("Stored IBKR credential data is invalid")
        return value

    def _write_json(self, secret_id: str, value: dict[str, Any]) -> None:
        client = self._get_client()
        body = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
        try:
            client.add_secret_version(
                request={
                    "parent": self._secret_name(secret_id),
                    "payload": {"data": body},
                }
            )
        except Exception as exc:
            raise IBKRStorageUnavailable("IBKR protected credential storage could not be updated") from exc

    def load(self) -> IBKRCredentials | None:
        client_info = self._read_json(self.client_secret_id)
        tokens = self._read_json(self.token_secret_id)
        if client_info is None or tokens is None:
            return None
        credentials = IBKRCredentials(client_info=client_info, tokens=tokens)
        _require_exact_read_scope(credentials)
        return credentials

    def persist(self, credentials: IBKRCredentials) -> None:
        _require_exact_read_scope(credentials)
        if credentials.client_info:
            prior_client_info = self._read_json(self.client_secret_id)
            if prior_client_info != credentials.client_info:
                self._write_json(self.client_secret_id, credentials.client_info)
        self._write_json(self.token_secret_id, credentials.tokens)


def _require_exact_read_scope(credentials: IBKRCredentials) -> None:
    scope = credentials.scope
    if not scope or set(scope.split()) != {READ_ONLY_SCOPE}:
        raise IBKRScopeRejected("IBKR grant must contain exactly the mcp.read scope")
    client_scope = credentials.client_info.get("scope")
    if client_scope and set(str(client_scope).split()) != {READ_ONLY_SCOPE}:
        raise IBKRScopeRejected("IBKR client registration exceeds the mcp.read scope")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_action_arguments(action: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate and copy the exact public input shape for each action."""
    allowed = {
        "get_symbol_daily_data": {"symbol", "trading_date"},
        "list_option_contracts": {"symbol", "expiry", "pagination"},
        "get_option_data": {"exact_contract_id", "symbol", "right", "strike", "expiry"},
    }.get(action)
    if allowed is None:
        raise ValueError("unsupported IBKR data action")
    if set(arguments) - allowed:
        raise ValueError("unexpected IBKR data action arguments")
    result = dict(arguments)
    if action == "get_symbol_daily_data":
        if not all(isinstance(result.get(key), str) and result[key].strip() for key in ("symbol", "trading_date")):
            raise ValueError("symbol and trading_date are required")
    elif action == "list_option_contracts":
        if not isinstance(result.get("symbol"), str) or not result["symbol"].strip():
            raise ValueError("symbol is required")
        if result.get("expiry") is not None and not isinstance(result["expiry"], str):
            raise ValueError("expiry must be a string")
        pagination = result.get("pagination", {})
        if not isinstance(pagination, dict) or set(pagination) - {"cursor", "limit"}:
            raise ValueError("pagination accepts only cursor and limit")
        if "limit" in pagination and (not isinstance(pagination["limit"], int) or not 1 <= pagination["limit"] <= 100):
            raise ValueError("pagination limit must be between 1 and 100")
        result["pagination"] = dict(pagination)
    else:
        exact_id = result.get("exact_contract_id")
        tuple_fields = ("symbol", "right", "strike", "expiry")
        has_any_tuple_field = any(key in result for key in tuple_fields)
        has_tuple = all(key in result for key in tuple_fields)
        if "exact_contract_id" in result and (not exact_id or has_any_tuple_field):
            raise ValueError("provide exact_contract_id alone or the complete exact contract tuple")
        if not exact_id and not has_tuple:
            raise ValueError("provide exact_contract_id or symbol, right, strike, and expiry")
        if exact_id and (isinstance(exact_id, bool) or not isinstance(exact_id, (str, int))):
            raise ValueError("exact_contract_id must be a string or integer")
        if has_tuple:
            if not all(isinstance(result[key], str) and result[key].strip() for key in ("symbol", "right", "expiry")):
                raise ValueError("symbol, right, and expiry must be non-empty strings")
            right = str(result["right"]).upper()
            if right not in {"C", "P", "CALL", "PUT"}:
                raise ValueError("right must identify a call or put")
            result["right"] = right
            if isinstance(result["strike"], bool) or not isinstance(result["strike"], (int, float)):
                raise ValueError("strike must be numeric")
    return result


_DAILY_FIELDS = ("open", "high", "low", "close", "volume")
_OPTION_FIELDS = ("price", "bid", "ask", "iv", "volume")


def _quality(value: Any) -> str:
    return value if value in {"live", "delayed"} else "unavailable"


def _observation_time(value: Any) -> str | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and len(value) <= 64:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc).isoformat()


def _available_fields(payload: dict[str, Any], names: tuple[str, ...]) -> tuple[dict[str, Any], list[str]]:
    source = payload.get("fields")
    if not isinstance(source, dict):
        source = {}
    values: dict[str, Any] = {}
    missing = []
    for name in names:
        value = source.get(name)
        if isinstance(value, bool) or (value is not None and not isinstance(value, (int, float))):
            value = None
        if isinstance(value, float) and not math.isfinite(value):
            value = None
        values[name] = value
        if name not in source or value is None:
            missing.append(name)
    return values, missing


def normalize_provider_result(action: str, arguments: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Project provider data onto the small public contract; drop raw fields."""
    retrieved_at = _utc_now()
    base = {
        "source": "IBKR",
        "observation_time": _observation_time(payload.get("observation_time")),
        "retrieval_time": retrieved_at,
        "quality": _quality(payload.get("quality")),
        "units": {},
    }
    raw_units = payload.get("units")
    if isinstance(raw_units, dict):
        allowed_unit_fields = _DAILY_FIELDS if action == "get_symbol_daily_data" else _OPTION_FIELDS
        base["units"] = {
            key: str(raw_units[key])[:80]
            for key in allowed_unit_fields
            if key in raw_units and isinstance(raw_units[key], str)
        }

    if action == "get_symbol_daily_data":
        identity_matches = (
            payload.get("data_status") != "missing"
            and str(payload.get("symbol", "")).upper() == str(arguments["symbol"]).upper()
            and str(payload.get("trading_date", "")) == str(arguments["trading_date"])
        )
        fields, unavailable = _available_fields(payload, _DAILY_FIELDS)
        if not identity_matches:
            fields = {name: None for name in _DAILY_FIELDS}
            unavailable = list(_DAILY_FIELDS)
        available = [name for name in _DAILY_FIELDS if name not in unavailable]
        return {**base, "symbol": arguments["symbol"], "trading_date": arguments["trading_date"],
                "data_status": "missing" if payload.get("data_status") == "missing" else ("found" if identity_matches else "unverified"),
                "fields": fields, "available_fields": available, "unavailable_fields": unavailable}

    if action == "list_option_contracts":
        rows = payload.get("contracts")
        contracts = []
        contract_fields = ("exact_contract_id", "right", "strike", "expiry")
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                contract = {key: row.get(key) for key in contract_fields}
                if isinstance(contract["exact_contract_id"], bool) or not isinstance(contract["exact_contract_id"], (str, int)):
                    contract["exact_contract_id"] = None
                if str(contract["right"]).upper() not in {"C", "P", "CALL", "PUT"}:
                    contract["right"] = None
                if isinstance(contract["strike"], bool) or not isinstance(contract["strike"], (int, float)):
                    contract["strike"] = None
                elif isinstance(contract["strike"], float) and not math.isfinite(contract["strike"]):
                    contract["strike"] = None
                if not isinstance(contract["expiry"], str):
                    contract["expiry"] = None
                requested_expiry = arguments.get("expiry")
                if requested_expiry and contract["expiry"] != requested_expiry:
                    continue
                contract["available_fields"] = [key for key in contract_fields if contract[key] is not None]
                contract["unavailable_fields"] = [key for key in contract_fields if contract[key] is None]
                contracts.append(contract)
        pagination = payload.get("pagination")
        safe_page = {}
        if isinstance(pagination, dict):
            if "next_cursor" in pagination and isinstance(pagination["next_cursor"], str):
                safe_page["next_cursor"] = pagination["next_cursor"]
            if "has_more" in pagination and isinstance(pagination["has_more"], bool):
                safe_page["has_more"] = pagination["has_more"]
        result = {**base, "symbol": arguments["symbol"], "contracts": contracts, "pagination": safe_page,
                  "contract_status": "found" if contracts else "missing"}
        if arguments.get("expiry"):
            result["requested_expiry"] = arguments["expiry"]
        return result

    missing_contract = payload.get("contract_status") == "missing"
    fields, unavailable = _available_fields({} if missing_contract else payload, _OPTION_FIELDS)
    actual_id = payload.get("exact_contract_id")
    expected_id = arguments.get("exact_contract_id")
    if expected_id is not None and actual_id is not None and str(actual_id) != str(expected_id):
        missing_contract = True
    identity_verified = actual_id is not None and expected_id is not None and str(actual_id) == str(expected_id)
    if not missing_contract and "symbol" in arguments:
        identity_verified = True
        for key in ("symbol", "right", "strike", "expiry"):
            actual = payload.get(key)
            expected = arguments[key]
            if actual is None:
                identity_verified = False
            elif str(actual).upper() != str(expected).upper():
                missing_contract = True
                break
    if missing_contract or not identity_verified:
        fields = {name: None for name in _OPTION_FIELDS}
        unavailable = list(_OPTION_FIELDS)
    contract_status = "missing" if missing_contract else ("found" if identity_verified else "unverified")
    result = {**base, "contract_status": contract_status,
              "fields": fields, "available_fields": [name for name in _OPTION_FIELDS if name not in unavailable],
              "unavailable_fields": unavailable}
    if contract_status != "found":
        result["requested_contract"] = (
            {"exact_contract_id": expected_id}
            if expected_id is not None
            else {key: arguments[key] for key in ("symbol", "right", "strike", "expiry")}
        )
    elif expected_id is not None:
        result["exact_contract_id"] = expected_id
    elif actual_id is not None:
        result["exact_contract_id"] = actual_id
    if contract_status == "found":
        for key in ("symbol", "right", "strike", "expiry"):
            value = arguments.get(key, payload.get(key))
            if value is not None:
                result[key] = value
    return result
