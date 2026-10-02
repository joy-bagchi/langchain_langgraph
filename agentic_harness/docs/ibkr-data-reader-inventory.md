# IBKR data reader integration inventory

Reviewed 2026-10-01. The live OAuth initiation and provider mapping remain
disabled until Harness ownership and the authenticated IBKR tool catalog are
verified.

## Existing components and disposition

| Codebase | Component | Disposition | Reason |
|---|---|---|---|
| Market-Physics-Core | `applications/ibkr_oauth_bootstrap.py` and `cloudbuild.ibkr-oauth-bootstrap.yaml` | Left disabled; callback/bootstrap code is not moved in this slice | Core's deployed initiation is disabled. Its last recorded authenticated initiation failed during MCP session initialization, so no client/token secret versions exist. No Cloud Run changes were made. |
| Market-Physics-Core | `adapters/auth/secret_manager_oauth.py` | Scope and versioning behavior reused in Harness | Harness now has its own GCP Secret Manager credential store that accepts exactly `mcp.read` and writes new token versions. No code or authentication implementation was added to Core. |
| Market-Physics-Core | `adapters/market_data/ibkr_tws.py` | Left outside this tool and disabled for this route | This is a separate socket API adapter, not the OAuth MCP connection. It cannot establish authenticated MCP tool schemas. |
| Market-Physics-Core | OAuth scope/bootstrap tests and deployment notes | Kept as reference; not copied as live evidence | They establish fail-closed scope and callback patterns, but do not prove current IBKR dynamic registration, consent, or market-data tool support. |
| Market-Physics-Core | `agents/mpf_agent/tools.yaml` | Left unchanged | It currently declares no IBKR action. Core is outside this writable checkout; its MPF agent still needs an explicit allowlist update before it can call the Harness actions. |
| Agentic Harness | `RegisteredToolService` `ibkr_data_pipeline` | Replaced in the default registry by three exact actions | The old tool accepted broad snapshot parameters and exposed Greeks/history behavior beyond this contract. Its ID is no longer registered. |
| Agentic Harness | workflow ledger, checkpoint, and resume path | Reused and extended | `authorization_required` is checkpointed. Resume clears market-data outputs from the affected work segment and reruns from its first IBKR action. |
| Agentic Harness | notifications | New service port; durable event is the fallback inbox | No notification service existed. Deployments can inject a delivery handler; run state records the authorization-required event even without one. |
| Agentic Harness | OAuth connection lifecycle | Harness-owned credential/provider boundary added; consent callback and live SDK transport remain unimplemented | Credentials load only in trusted tool code. The provider must refresh first and return rotated credentials for versioned persistence. A concrete live MCP adapter cannot be mapped before authenticated tool schemas are inspected. |
| `langchain_langgraph` | `ibkr_market_data_agent` workflow | Wired to the three Harness actions | Its inputs now include the trading date and an explicit exact contract ID. The Harness does not choose a substitute contract. |
| `langchain_langgraph` | daily regime `ibkr_data_pipeline` use | Left disabled | Those workflows require multi-symbol, multi-day history and snapshot fields not present in the three-action contract. Reusing the local TWS/client snapshot here would bypass Harness OAuth ownership and would assume provider capabilities that have not been inspected. |
| `langchain_langgraph` | local `agentic_vol_regime_app.data.ibkr_client` and ingestion utilities | Left in place for existing non-Harness local workflows | They are standalone TWS socket/history consumers. They are not used as the Harness provider and do not duplicate OAuth or token storage. |
| `langchain_langgraph` | `agentic_ibkr_account_app` account, order, and option-chain UI | Left separate from the model-facing registry | Account/order surfaces are outside this read-only quote contract and must not become model-visible tools. |

## Harness contract

The model-visible action set is exactly:

- `get_symbol_daily_data(symbol, trading_date)`
- `list_option_contracts(symbol, expiry?, pagination?)`
- `get_option_data(exact_contract_id | symbol, right, strike, expiry)`

Provider output is projected onto the action's allowlisted fields. Missing
contracts and unavailable fields are explicit. Results include `source`,
`observation_time`, `retrieval_time`, `units`, and `quality`. Raw MCP catalogs,
provider exceptions, account data, write tools, and credentials are not
returned.

The credential store uses Application Default Credentials and Secret Manager
inside Harness runtime code. Configure separate secret IDs through
`IBKR_OAUTH_CLIENT_SECRET_ID` and `IBKR_OAUTH_TOKEN_SECRET_ID`, and the project
through `GOOGLE_CLOUD_PROJECT`. Runtime IAM should allow access to the client
and token secrets and adding token versions only. No credential belongs in
agent YAML or tool arguments.

The provider interface calls `refresh_credentials` before each data action.
An expired/consent-required grant must raise the safe
`IBKRAuthorizationRequired` signal; a successful rotation is scope-checked and
persisted before data invocation. Broader or missing scopes fail closed.

## Live blocker

No authenticated IBKR MCP tool catalog was available in this workspace or
connected tool session. Core's current deployment notes also report OAuth
initiation disabled and no persisted credential versions. Therefore the
provider tool names, JSON schemas, historical data support, IV support, and
read-only grant have not been verified. No provider calls are guessed or
configured here.

The next live step is to move the Core callback/state handling into a disabled
Harness-owned bootstrap, diagnose the recorded MCP session-initialization
failure, and verify secret handling. Then initiate exact `mcp.read` consent and
inspect authenticated `list_tools` schemas inside trusted runtime code. Add one reviewed mapping for
the required quote action only after confirming its contract ID, quote fields,
observation time, and live/delayed quality behavior. Keep Core Cloud Run
initiation disabled during that verification.
