# POMDP_VOL_REGIME_APP.md

# Agentic Volatility Regime POMDP Application

## Purpose

Build an agentic application on top of `agentic_harness` / Agentic OS that performs volatility-regime inference, regime-shift prediction, policy recommendation, backtesting, alerting, and long-term learning for a Deep ITM LEAP + selective short-call overwrite strategy.

The application should behave as a **Partially Observable Markov Decision Process (POMDP)** system:

- The true volatility regime is hidden.
- The system observes noisy market and options signals.
- The system maintains a probabilistic belief over hidden regimes.
- The system updates beliefs as new observations arrive.
- The system predicts likely regime transitions.
- The system recommends strategy posture.
- The system learns from backtests and live outcomes.
- The system stores validated learnings in durable and semantic memory.

The application is not intended to directly trade at first. It should begin as a decision-support and alerting system.

---

## Strategic Context

The trading strategy is:

- Long Deep ITM LEAP call as core directional exposure.
- Sell short-dated calls selectively against the LEAP.
- Avoid overwriting in persistent low-volatility bull regimes.
- Increase overwriting when volatility regime instability rises.
- Detect early warnings of volatility expansion before VIX fully spikes.
- Reduce or adapt overwriting after panic-vol compression and rebound risk.

The central modeling problem is not simply:

> Will SPY go up or down?

The correct problem is:

> Given noisy observations, what is the current hidden volatility regime, how likely is it to transition, and what action is best under that belief distribution?

---

## Agentic OS Fit

This application should use the existing Agentic OS capabilities:

- Runtime ledger for runs, checkpoints, events, artifacts, memory, and invocations.
- Postgres runtime ledger for durable production-grade persistence.
- Local JSON mirrors for inspectability.
- LangSmith tracing for observability.
- YAML-defined agents.
- Workflow-backed agent execution.
- Markdown workflow DSL.
- Declarative DAG workflow definitions.
- DAG execution with stage-level parallelism.
- Human review gates.
- Durable memory.
- Structured memory.
- Semantic memory with pgvector.
- Context manager and compaction.
- Guardrails.
- Critic-aware evaluation engine.
- Separate internal, artifact, and response output channels.

The POMDP volatility application should build on these platform features rather than recreate them.

---

# 1. Core POMDP Model

## 1.1 Hidden State

The true hidden state is the latent market-volatility regime.

Initial regime set:

```text
STABLE_LOW_VOL_TREND
MID_VOL_CHOP
VOL_EXPANSION_TRANSITION
HIGH_VOL_RISK_OFF
PANIC_CONVEXITY_STRESS
POST_PANIC_COMPRESSION
```

These are not directly observable. They are inferred.

## 1.2 Observations

Observed variables may include:

### Market Price

- SPY / SPX returns
- QQQ returns
- rolling trend
- drawdown
- moving average structure
- breadth proxies if available

### Realized Volatility

- 5d realized volatility
- 10d realized volatility
- 21d realized volatility
- 63d realized volatility
- realized volatility acceleration
- short-vol / long-vol ratio

### Implied Volatility

- VIX
- VIX z-score
- VIX percentile
- VIX change
- VIX acceleration

### Vol-of-Vol

- VVIX
- VVIX z-score
- VVIX percentile
- VVIX/VIX
- VVIX/VIX z-score
- VVIX/VIX trend
- VVIX/VIX acceleration

### Term Structure

- VIX9D / VIX
- VIX / VIX3M
- VIX3M - VIX
- contango / backwardation state
- term structure flattening speed

### Options Surface / IBKR Data

- short call IV
- LEAP IV
- front IV / LEAP IV
- delta buckets
- skew
- call premium richness
- bid/ask width
- open interest and liquidity
- strike availability

### Optional Experimental Features

- VOLI
- VOLI/VIX
- skew index
- put-call ratio
- credit spread proxy
- MOVE
- DXY

Experimental features should be ingested but not manually over-weighted until backtests prove incremental value.

## 1.3 Belief State

The belief state is a probability vector:

```json
{
  "as_of": "YYYY-MM-DDTHH:MM:SSZ",
  "beliefs": {
    "STABLE_LOW_VOL_TREND": 0.62,
    "MID_VOL_CHOP": 0.18,
    "VOL_EXPANSION_TRANSITION": 0.12,
    "HIGH_VOL_RISK_OFF": 0.06,
    "PANIC_CONVEXITY_STRESS": 0.01,
    "POST_PANIC_COMPRESSION": 0.01
  },
  "entropy": 0.91,
  "model_confidence": 0.74
}
```

The belief state should be persisted after every update.

## 1.4 Actions

Initial decision-support actions:

```text
NO_OVERWRITE
LIGHT_OVERWRITE
MEDIUM_OVERWRITE
AGGRESSIVE_OVERWRITE
REDUCE_OVERWRITE
ROLL_SHORT_CALL
CLOSE_SHORT_CALL
HEDGE_OR_DE_RISK
MANUAL_REVIEW
```

The first version should recommend actions but not execute trades.

## 1.5 Reward

Backtesting reward should include:

```text
strategy_return
- drawdown_penalty
- upside_truncation_penalty
- missed_premium_penalty
- false_transition_alert_penalty
- missed_transition_penalty
- excessive_turnover_penalty
- transaction_cost_penalty
```

Reward must be configurable.

Do not optimize only for total return.

---

# 2. Agent Architecture

## 2.1 Data Ingestion Agent

### Responsibility

Fetch and normalize market and options data.

### Inputs

- IBKR API
- Yahoo/Stooq/CBOE fallback sources
- local cached data
- configured symbol universe

### Outputs

- raw market snapshots
- raw option-chain snapshots
- data quality report
- normalized observation records

### Requirements

- Must not silently overwrite data.
- Must persist raw observations.
- Must flag missing, stale, or inconsistent values.
- Must support replay from historical data.

---

## 2.2 Feature Engineering Agent

### Responsibility

Transform raw observations into model-ready features.

### Outputs

```json
{
  "as_of": "YYYY-MM-DD",
  "features": {
    "vix": 18.38,
    "vvix": 98.0,
    "vvix_vix_ratio": 5.34,
    "vvix_vix_z_22d": 0.26,
    "vix_21d_z": 0.42,
    "rv_21d": 0.147,
    "vix_rv_spread": 0.036,
    "vix3m_minus_vix": 1.8,
    "trend_persistence_21d": 0.63
  }
}
```

### Requirements

- All rolling features must avoid lookahead bias.
- Feature definitions must be deterministic.
- Feature schema must be versioned.
- Missing features should be explicit, not silently set to zero.

---

## 2.3 Belief State Agent

### Responsibility

Maintain and update the hidden-regime belief distribution.

### Initial Methods

Start simple:

1. heuristic prior
2. HMM filtered probabilities
3. Bayesian update from observation likelihoods
4. rolling transition matrix

Later:

- Bayesian HMM
- particle filter
- Markov-switching stochastic volatility
- ensemble belief model

### Outputs

- current belief vector
- belief delta from prior update
- confidence score
- entropy score
- top drivers

### Example

```json
{
  "previous_belief": {
    "STABLE_LOW_VOL_TREND": 0.71,
    "VOL_EXPANSION_TRANSITION": 0.10
  },
  "updated_belief": {
    "STABLE_LOW_VOL_TREND": 0.58,
    "VOL_EXPANSION_TRANSITION": 0.24
  },
  "belief_delta": {
    "STABLE_LOW_VOL_TREND": -0.13,
    "VOL_EXPANSION_TRANSITION": 0.14
  },
  "drivers": [
    "VVIX/VIX z-score rose above 1.5",
    "VIX term structure flattened",
    "realized volatility acceleration positive"
  ]
}
```

---

## 2.4 Transition Prediction Agent

### Responsibility

Estimate probability of near-term volatility regime shift.

### Horizons

- 1 trading day
- 3 trading days
- 5 trading days
- 10 trading days
- 21 trading days

### Target Events

```text
VOL_RISE
VOL_EXPANSION
VIX_SPIKE
VIX_EXPLOSION
RISK_OFF_TRANSITION
PANIC_CONVEXITY_STRESS
VOL_COMPRESSION
```

### Example Output

```json
{
  "as_of": "YYYY-MM-DD",
  "transition_probabilities": {
    "vol_expansion_5d": 0.38,
    "vix_spike_5d": 0.21,
    "vix_explosion_10d": 0.11,
    "risk_off_transition_10d": 0.29
  },
  "top_predictive_factors": [
    "VVIX/VIX z-score persistence",
    "term structure flattening",
    "realized volatility acceleration"
  ]
}
```

---

## 2.5 Predictive Alert Agent

### Responsibility

Generate early warnings when probability distributions imply volatility may rise sharply.

This agent should not simply alert because VIX is high.

It should alert when the underlying probability distribution suggests rising probability of a volatility transition.

### Alert Types

```text
WATCH
WARNING
HIGH_RISK
CRITICAL
```

### Predictive Alert Examples

#### WATCH

```text
Transition probability has risen but confirmation is weak.
```

#### WARNING

```text
Regime transition probability is materially elevated and multiple features confirm.
```

#### HIGH_RISK

```text
Probability of VIX spike over next 5-10 trading days is elevated.
Review overwrite and risk posture.
```

#### CRITICAL

```text
Convexity stress probability is extreme.
Manual review required before new overwrite exposure.
```

### Suggested Initial Thresholds

```yaml
watch:
  vol_expansion_5d: 0.30
  transition_belief: 0.25

warning:
  vol_expansion_5d: 0.45
  transition_belief: 0.35
  confirming_features_min: 2

high_risk:
  vix_spike_10d: 0.35
  transition_belief: 0.45
  high_vol_belief: 0.25

critical:
  panic_belief: 0.30
  vix_explosion_10d: 0.45
```

These thresholds are starting points only. They should be calibrated by backtesting.

### Alert Payload

```json
{
  "alert_id": "uuid",
  "as_of": "YYYY-MM-DDTHH:MM:SSZ",
  "severity": "WARNING",
  "alert_type": "PREDICTIVE_VOL_EXPANSION",
  "headline": "Volatility expansion risk is rising",
  "probabilities": {
    "vol_expansion_5d": 0.48,
    "vix_spike_10d": 0.31
  },
  "belief_state": {
    "VOL_EXPANSION_TRANSITION": 0.39,
    "HIGH_VOL_RISK_OFF": 0.18
  },
  "drivers": [
    "VVIX/VIX z-score > 1.5 for 3 sessions",
    "VIX/VIX3M term structure flattened",
    "realized volatility acceleration positive"
  ],
  "recommended_review": [
    "Avoid initiating tight short calls",
    "Review existing overwrite exposure",
    "Check short-call deltas and roll risk"
  ],
  "requires_human_review": true
}
```

---

## 2.6 Policy Recommendation Agent

### Responsibility

Map belief state and transition probabilities to strategy posture.

### Initial Policy Logic

```text
If STABLE_LOW_VOL_TREND high and transition risk low:
    recommend NO_OVERWRITE or LIGHT_OVERWRITE

If VOL_EXPANSION_TRANSITION rising:
    recommend LIGHT_OVERWRITE or MEDIUM_OVERWRITE

If HIGH_VOL_RISK_OFF high:
    recommend MEDIUM_OVERWRITE or AGGRESSIVE_OVERWRITE

If PANIC_CONVEXITY_STRESS high:
    recommend MANUAL_REVIEW, avoid naive new exposure

If POST_PANIC_COMPRESSION high:
    recommend REDUCE_OVERWRITE or avoid tight caps due to rebound risk
```

### Output

```json
{
  "recommended_action": "LIGHT_OVERWRITE",
  "confidence": 0.68,
  "rationale": [
    "Stable-low-vol belief remains dominant",
    "Transition probability is rising but below warning threshold",
    "VVIX/VIX trend is not yet confirming stress"
  ],
  "risk_notes": [
    "Avoid tight upside cap if trend persistence remains strong"
  ]
}
```

---

## 2.7 Backtest Agent

### Responsibility

Evaluate regime inference, prediction, alerts, and policy recommendations historically.

### Modes

1. historical replay
2. walk-forward validation
3. parameter sensitivity
4. alert quality evaluation
5. policy outcome comparison

### Required Baselines

```text
NEVER_OVERWRITE
ALWAYS_OVERWRITE
VIX_THRESHOLD_OVERWRITE
HEURISTIC_REGIME_OVERWRITE
POMDP_BELIEF_POLICY
```

### Metrics

- annualized return
- Sharpe
- Sortino
- max drawdown
- Calmar
- upside capture
- downside capture
- overwrite income
- upside truncation cost
- false alert rate
- missed transition rate
- average alert lead time
- precision / recall for vol expansion
- policy turnover
- transaction cost sensitivity

---

## 2.8 Learning and Memory Agent

### Responsibility

Store validated observations and lessons in long-term memory.

### Important Rule

Do not store every backtest result as a trusted lesson.

Use memory promotion stages.

```text
candidate_memory
validated_memory
promoted_model_prior
deprecated_memory
```

### Candidate Memory Example

```json
{
  "memory_type": "candidate_signal_lesson",
  "signal": "VVIX/VIX z-score persistence",
  "condition": "z_22d > 1.5 for at least 3 sessions",
  "target": "vol_expansion_5d",
  "result": "improved recall but increased false positives",
  "sample_period": "2018-01-01 to 2024-12-31",
  "status": "candidate_memory"
}
```

### Promoted Memory Example

```json
{
  "memory_type": "promoted_signal_lesson",
  "signal": "VVIX/VIX z-score + term structure flattening",
  "condition": "z_22d > 1.5 and VIX/VIX3M rising",
  "target": "vol_expansion_10d",
  "result": "improved transition prediction across walk-forward periods",
  "status": "promoted_model_prior",
  "promotion_evidence": [
    "positive out-of-sample precision improvement",
    "stable across 2018, 2020, 2022, 2025 regimes"
  ]
}
```

### Requirements

- Memory must be structured and searchable.
- Memory should include sample period and validation method.
- Memory should include negative lessons.
- Memory should support deprecation.
- Memory should not silently alter model weights without review.

---

## 2.9 Evaluation / Critic Agent

### Responsibility

Critique model outputs, policy recommendations, and memory promotions.

### Checks

- Is the alert based on one noisy feature or multiple confirming signals?
- Is the model overreacting to short-term noise?
- Did the belief update conflict with known regime context?
- Is the recommendation consistent with current position risk?
- Is this a candidate learning or a robust learning?
- Did the model overfit the latest backtest?

### Possible Actions

```text
ALLOW
RETRY
ESCALATE_TO_HUMAN
FAIL
MARK_AS_CANDIDATE_ONLY
```

---

# 3. Workflows

## 3.1 Daily Regime Update Workflow

```text
1. ingest_market_data
2. validate_data_quality
3. compute_features
4. update_belief_state
5. estimate_transition_probabilities
6. generate_alerts
7. recommend_policy
8. critic_review
9. persist_artifacts
10. write_memory_candidates
11. produce_daily_report
```

### Output Artifact

`reports/daily_regime_report_YYYY-MM-DD.md`

---

## 3.2 Intraday Monitoring Workflow

This is optional after daily workflow is stable.

```text
1. ingest_intraday_snapshot
2. compute_intraday_features
3. compare_to_daily_belief
4. update_short_horizon_belief_delta
5. detect_predictive_alerts
6. escalate if threshold crossed
```

Important:

- Intraday data is noisier.
- Intraday alerts should require persistence or confirmation.
- Avoid flickering alerts.

---

## 3.3 Historical Replay Workflow

```text
1. select_historical_period
2. replay observations in chronological order
3. update beliefs step-by-step
4. generate simulated alerts
5. record recommended policy
6. score predictions and policy outcomes
7. produce replay report
```

---

## 3.4 Backtest Learning Workflow

```text
1. run walk-forward backtest
2. evaluate alert quality
3. evaluate policy performance
4. identify signal failures and successes
5. create candidate memories
6. critic reviews candidate memories
7. promote only robust lessons
8. persist backtest artifact
```

---

## 3.5 Human Review Workflow

Human review is required for:

- memory promotion to model prior
- changing production alert thresholds
- changing policy action mapping
- enabling new feature family
- enabling trade execution in future versions

---

# 4. Data Contracts

## 4.1 Observation Record

```json
{
  "schema_version": "observation.v1",
  "as_of": "YYYY-MM-DDTHH:MM:SSZ",
  "source": "IBKR",
  "symbols": {
    "SPY": {
      "last": 0.0,
      "close": 0.0,
      "volume": 0
    },
    "VIX": {
      "last": 0.0
    },
    "VVIX": {
      "last": 0.0
    }
  },
  "quality": {
    "is_complete": true,
    "stale_fields": [],
    "warnings": []
  }
}
```

## 4.2 Feature Record

```json
{
  "schema_version": "features.v1",
  "as_of": "YYYY-MM-DDTHH:MM:SSZ",
  "feature_set_version": "vol_regime_features_v1",
  "features": {},
  "missing_features": [],
  "lookback_windows": {
    "zscore_short": 22,
    "rv_medium": 21,
    "rv_long": 63
  }
}
```

## 4.3 Belief Record

```json
{
  "schema_version": "belief.v1",
  "as_of": "YYYY-MM-DDTHH:MM:SSZ",
  "model_version": "belief_model_v1",
  "beliefs": {},
  "belief_delta": {},
  "entropy": 0.0,
  "confidence": 0.0,
  "drivers": []
}
```

## 4.4 Alert Record

```json
{
  "schema_version": "alert.v1",
  "alert_id": "uuid",
  "as_of": "YYYY-MM-DDTHH:MM:SSZ",
  "severity": "WATCH",
  "alert_type": "PREDICTIVE_VOL_EXPANSION",
  "probabilities": {},
  "drivers": [],
  "recommended_review": [],
  "requires_human_review": false
}
```

## 4.5 Policy Recommendation Record

```json
{
  "schema_version": "policy_recommendation.v1",
  "as_of": "YYYY-MM-DDTHH:MM:SSZ",
  "recommended_action": "NO_OVERWRITE",
  "confidence": 0.0,
  "rationale": [],
  "risk_notes": [],
  "requires_human_review": false
}
```

---

# 5. Suggested Repository Structure

```text
agentic_vol_regime_app/
├── README.md
├── POMDP_VOL_REGIME_APP.md
├── configs/
│   ├── agents/
│   │   ├── data_ingestion_agent.yaml
│   │   ├── feature_engineering_agent.yaml
│   │   ├── belief_state_agent.yaml
│   │   ├── transition_prediction_agent.yaml
│   │   ├── predictive_alert_agent.yaml
│   │   ├── policy_recommendation_agent.yaml
│   │   ├── backtest_agent.yaml
│   │   ├── memory_learning_agent.yaml
│   │   └── critic_agent.yaml
│   ├── workflows/
│   │   ├── daily_regime_update.yaml
│   │   ├── intraday_monitoring.yaml
│   │   ├── historical_replay.yaml
│   │   └── backtest_learning.yaml
│   ├── thresholds/
│   │   └── alert_thresholds.yaml
│   └── features/
│       └── feature_set_v1.yaml
├── src/
│   ├── data/
│   │   ├── ibkr_client.py
│   │   ├── market_data_loader.py
│   │   ├── option_chain_loader.py
│   │   └── quality.py
│   ├── features/
│   │   ├── realized_vol.py
│   │   ├── implied_vol.py
│   │   ├── term_structure.py
│   │   ├── trend.py
│   │   └── build_features.py
│   ├── pomdp/
│   │   ├── states.py
│   │   ├── observations.py
│   │   ├── belief_update.py
│   │   ├── transition_model.py
│   │   ├── rewards.py
│   │   └── policy.py
│   ├── alerts/
│   │   ├── rules.py
│   │   ├── predictive_alerts.py
│   │   └── alert_renderer.py
│   ├── backtest/
│   │   ├── replay.py
│   │   ├── policies.py
│   │   ├── metrics.py
│   │   └── walk_forward.py
│   ├── memory/
│   │   ├── schemas.py
│   │   ├── promotion.py
│   │   └── lessons.py
│   └── reports/
│       ├── daily_report.py
│       └── backtest_report.py
├── tests/
│   ├── test_feature_no_lookahead.py
│   ├── test_belief_update.py
│   ├── test_alert_thresholds.py
│   ├── test_replay_chronological.py
│   └── test_memory_promotion.py
└── reports/
    ├── daily/
    ├── backtests/
    └── alerts/
```

---

# 6. First Implementation Milestones

## Milestone 1: Daily Belief Report

Build a daily workflow that produces:

- data quality status
- feature snapshot
- current regime belief vector
- transition probabilities
- alert status
- policy recommendation
- markdown report

No backtesting yet.

## Milestone 2: Historical Replay

Replay historical data one day at a time and persist belief states.

Measure whether alert signals occurred before known VIX spikes.

## Milestone 3: Alert Calibration

Tune thresholds using historical replay.

Track:

- false positives
- missed transitions
- lead time
- persistence requirements

## Milestone 4: Policy Backtest

Compare:

- never overwrite
- always overwrite
- heuristic overwrite
- POMDP belief policy

## Milestone 5: Memory Learning

Store validated signal lessons.

Add critic review and human promotion gates.

## Milestone 6: Intraday Monitoring

Add intraday snapshots only after daily system is stable.

Use persistence requirements to avoid noise.

---

# 7. Guardrails

## 7.1 No Unreviewed Live Trading

The system must not place live trades in v1.

## 7.2 No Silent Model Mutation

The system must not change model priors, thresholds, or production policy mappings without review.

## 7.3 No Lookahead Bias

Backtesting must replay observations chronologically.

## 7.4 No Single-Feature Critical Alerts

Critical predictive alerts should require multiple confirming signals.

## 7.5 Memory Promotion Requires Evidence

Backtest findings start as candidate memories.

Only robust findings become promoted model priors.

---

# 8. Predictive VIX Explosion Detection

## 8.1 Definition

A VIX explosion event should be explicitly defined.

Initial possible definitions:

```text
VIX_EXPLOSION_5D:
  VIX rises by >= 30% within 5 trading days

VIX_SPIKE_10D:
  VIX rises by >= 40% within 10 trading days

HIGH_VOL_BREAKOUT:
  VIX crosses above 25 from below 20 within 10 trading days

CONVEXITY_STRESS:
  VIX rises, VVIX rises, and term structure flattens/backwardates
```

These labels are for training and evaluation only.

## 8.2 Candidate Predictive Signals

- VVIX/VIX z-score spike
- VVIX/VIX z-score persistence
- VVIX acceleration
- VIX acceleration
- VIX9D/VIX rising
- VIX/VIX3M rising
- VIX term structure flattening
- realized volatility acceleration
- SPY trend breakdown
- drawdown acceleration
- IV/RV spread expansion
- short-term option premium richness

## 8.3 Predictive Alert Logic

A predictive alert should combine:

```text
probability estimate
+ persistence
+ confirmation count
+ model confidence
+ current position risk
```

Example:

```text
If:
  P(VIX_EXPLOSION_10D) > 0.35
  and VOL_EXPANSION_TRANSITION belief > 0.40
  and at least 2 confirming features are active
Then:
  issue HIGH_RISK predictive alert
```

---

# 9. Daily Report Template

```markdown
# Daily Volatility Regime Report

Date: YYYY-MM-DD

## Summary

Current regime belief favors: `<REGIME>`

Transition risk: `<LOW | MODERATE | HIGH | CRITICAL>`

Recommended posture: `<ACTION>`

## Belief State

| Regime | Probability |
|---|---:|
| Stable Low-Vol Trend | 0.00 |
| Mid-Vol Chop | 0.00 |
| Vol Expansion Transition | 0.00 |
| High-Vol Risk-Off | 0.00 |
| Panic Convexity Stress | 0.00 |
| Post-Panic Compression | 0.00 |

## Key Signals

| Signal | Value | Interpretation |
|---|---:|---|
| VIX | 0.00 |  |
| VVIX | 0.00 |  |
| VVIX/VIX | 0.00 |  |
| VVIX/VIX z-score | 0.00 |  |
| VIX term structure |  |  |
| Realized vol trend |  |  |

## Predictive Alerts

None / WATCH / WARNING / HIGH_RISK / CRITICAL

## Policy Recommendation

Recommended action: `<ACTION>`

Rationale:

- item
- item
- item

## Model Confidence

Confidence: 0.00

Uncertainty / entropy: 0.00

## Required Human Review

Yes / No
```

---

# 10. Codex First Task

Ask Codex to begin with:

```text
Create the repository scaffold and implement Milestone 1: Daily Belief Report.

Use deterministic placeholder models first:
- feature calculation
- heuristic belief update
- simple transition probability function
- alert threshold rules
- markdown report generation

Do not implement ML yet.
Do not implement live trading.
Do not implement automatic memory promotion.
```

---

# 11. Development Philosophy

Build in this order:

1. Observability
2. Deterministic features
3. Heuristic belief update
4. Daily report
5. Historical replay
6. Alert calibration
7. HMM/Bayesian inference
8. Backtesting
9. Memory learning
10. Intraday monitoring
11. Policy optimization
12. Optional RL

The system should become intelligent gradually.

The first version should be simple, inspectable, and hard to fool.
