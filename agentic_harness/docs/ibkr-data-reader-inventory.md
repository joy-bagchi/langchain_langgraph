# IBKR data reader integration inventory

Reviewed 2026-10-02. The Harness OAuth bootstrap is implemented and its offline
contract tests pass against the pinned MCP SDK fork. One Harness-initiated
registration attempt reached IBKR's advertised `/oauth2/register` endpoint and
received HTTP 403; no consent link was returned. Provider action mapping and
MPF exposure remain disabled.

## Existing components and disposition

| Codebase | Component | Disposition | Reason |
|---|---|---|---|
| Market-Physics-Core | `applications/ibkr_oauth_bootstrap.py` and `cloudbuild.ibkr-oauth-bootstrap.yaml` | Left as disabled source reference; safe discovery, DCR checks, callback-state validation, and diagnostics adapted under Harness ownership | Its dedicated Cloud Run service was deleted before Harness initiation. Harness uses its own app, callback, runtime identities, and Secret Manager IDs. Other MPF services were not changed. |
| Market-Physics-Core | `adapters/auth/secret_manager_oauth.py` | Replaced by Harness-native Secret Manager storage and MCP token-storage adapter | Only exact `mcp.read` tokens are accepted; token/client records use separate versioned secrets under the Harness runtime identity. Core code is not imported at runtime. |
| Market-Physics-Core | `adapters/market_data/ibkr_tws.py` | Left outside this tool and disabled for this route | This is a separate socket API adapter, not the OAuth MCP connection. It cannot establish authenticated MCP tool schemas. |
| Market-Physics-Core | OAuth scope/bootstrap tests and deployment notes | Kept as reference; not copied as live evidence | They establish fail-closed scope and callback patterns, but do not prove current IBKR dynamic registration, consent, or market-data tool support. |
| Market-Physics-Core | `agents/mpf_agent/tools.yaml` | Left unchanged | It currently declares no IBKR action. Core is outside this writable checkout; its MPF agent still needs an explicit allowlist update before it can call the Harness actions. |
| Agentic Harness | `RegisteredToolService` IBKR actions | Three bounded actions are registered only when trusted runtime code explicitly injects a provider | The default toolbox exposes no IBKR actions. The list signature is `list_option_contracts(symbol, optional_expiry?, pagination?)`; other actions retain exact contract selection. |
| Agentic Harness | workflow ledger, checkpoint, and resume path | Reused and extended | `authorization_required` is checkpointed. Resume clears market-data outputs from the affected work segment and reruns from its first IBKR action. |
| Agentic Harness | notifications | New service port; durable event is the fallback inbox | No notification service existed. Deployments can inject a delivery handler; run state records the authorization-required event even without one. |
| Agentic Harness | OAuth connection lifecycle | Implemented in `agentic_harness/ibkr_oauth.py`; pinned MCP fork; callback and DCR endpoint discovered from IBKR metadata | Hosted service uses a one-instance callback process, identity-gated initiation, no-store link responses, one-use state, a callback request-log exclusion, and exact-scope Secret Manager storage. The single registration attempt failed with HTTP 403; no tool was listed or invoked. Initiation is disabled and the service is scaled to zero. |
| `langchain_langgraph` | Local MPF YAML action allowlists and workflows | Left unchanged; effective runtime exposure remains off while no trusted provider is configured | The default Harness registry has no IBKR action. Existing ChatGPT MPF connections were not modified. The catalog and supported data capabilities remain unverified. |
| `langchain_langgraph` | daily regime `ibkr_data_pipeline` use | Left disabled | Those workflows require multi-symbol, multi-day history and snapshot fields not present in the three-action contract. Reusing the local TWS/client snapshot here would bypass Harness OAuth ownership and would assume provider capabilities that have not been inspected. |
| `langchain_langgraph` | local `agentic_vol_regime_app.data.ibkr_client` and ingestion utilities | Left in place for existing non-Harness local workflows | They are standalone TWS socket/history consumers. They are not used as the Harness provider and do not duplicate OAuth or token storage. |
| `langchain_langgraph` | `agentic_ibkr_account_app` account, order, and option-chain UI | Left separate from the model-facing registry | Account/order surfaces are outside this read-only quote contract and must not become model-visible tools. |

## Harness contract

When a reviewed trusted provider is explicitly configured, the IBKR model-visible
action set is exactly:

- `get_symbol_daily_data(symbol, trading_date)`
- `list_option_contracts(symbol, optional_expiry?, pagination?)`
- `get_option_data(exact_contract_id | symbol, right, strike, expiry)`

Provider output is projected onto the action's allowlisted fields. Missing
contracts and unavailable fields are explicit. Results include `source`,
`observation_time`, `retrieval_time`, `units`, and `quality`. Raw MCP catalogs,
provider exceptions, account data, write tools, and credentials are not
returned.

The data-reader provider uses Application Default Credentials and Secret
Manager inside Harness runtime code. Its generic tool configuration reads
`IBKR_OAUTH_CLIENT_SECRET_ID` and `IBKR_OAUTH_TOKEN_SECRET_ID`; the hosted
OAuth service has separate `HARNESS_IBKR_CLIENT_SECRET_ID`,
`HARNESS_IBKR_TOKEN_SECRET_ID`, and `HARNESS_IBKR_TRANSACTION_SECRET_ID`
settings. Runtime IAM should allow reading the client, token, and transaction
secrets and adding token versions only. No credential belongs in agent YAML
or tool arguments.

The provider interface calls `refresh_credentials` before each data action. The
Harness MCP provider uses the pinned SDK to refresh expired grants when a
refresh token is supported. An expired or consent-required grant raises the
safe `IBKRAuthorizationRequired` signal; a successful rotation is scope-checked
and persisted before data invocation. Broader or missing scopes fail closed.

## Live blocker

No authenticated IBKR MCP tool catalog or completed Harness consent is available
yet. Therefore provider tool names, JSON schemas, historical data support, IV
support, and the live read-only grant have not been verified. The one DCR
attempt reported endpoint `https://api.ibkr.com/oauth2/register`, HTTP status
403, scope `mcp.read`, application type `web`, grant types
`authorization_code,refresh_token`, response type `code`, and the Harness
`/ibkr/callback` URI. The structured OAuth error code and description were not
available. No provider calls are guessed or configured here.

The live blocker is IBKR's unexplained 403 for the exact Harness DCR request.
The Harness initiator is disabled. The next step is an IBKR support response or
an externally documented registration-policy clarification; no retry or scope
change is configured. After an authorized registration and completed consent,
inspect authenticated `list_tools` schemas inside trusted runtime code before
adding any reviewed data mapping.
