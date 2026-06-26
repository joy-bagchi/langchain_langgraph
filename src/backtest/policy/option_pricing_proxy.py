from __future__ import annotations

import math


def _norm_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))


def black_scholes_call_price(
    *,
    spot: float,
    strike: float,
    dte: int,
    iv_annual: float,
    risk_free_rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> float:
    s = max(float(spot), 1e-6)
    k = max(float(strike), 1e-6)
    t = max(float(dte) / 252.0, 1e-9)
    sigma = max(float(iv_annual), 1e-6)
    r = float(risk_free_rate)
    q = float(dividend_yield)

    if sigma <= 1e-6 or t <= 1e-9:
        return max(s - k, 0.0)

    sqrt_t = math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    call_price = (s * math.exp(-q * t) * _norm_cdf(d1)) - (k * math.exp(-r * t) * _norm_cdf(d2))
    return max(float(call_price), max(s - k, 0.0), 0.0)

