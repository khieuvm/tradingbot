---
name: regime-detector
description: Detect VN30F1M intraday market regime using ADX, ATR, and DI+/DI- from 15m timeframe. Classify as TRENDING, RANGING, VOLATILE, or NORMAL. Use when user asks about current market conditions, regime, whether to trade, or when filtering signals by regime. Determines which combo categories are active. Intraday only — sessions 9:00-11:30 and 13:00-14:30 (UTC+7).
---

# VN30F1M Regime Detector

Detect the current intraday market regime for VN30F1M futures to determine which trading strategies should be active.

## Session Context (Critical)

VN30F1M trades in TWO separate intraday sessions (Vietnam time, UTC+7):

| Session | Time | Characteristics |
|---------|------|-----------------|
| Morning (AM) | 9:00–11:30 | Higher volume, ORB setups, trend establishment |
| Afternoon (PM) | 13:00–14:30 | Lower volume, mean-reversion more common, EOD flatten |
| Lunch break | 11:30–13:00 | NO TRADING — market closed |
| Pre-close | 14:15–14:28 | Close all positions, no new entries |

**Intraday rules:**
- NO overnight holds — all positions flatten by 14:28
- Regime resets at 13:00 (afternoon may differ from morning)
- First 15 min (9:00-9:15) — regime unreliable, wait for confirmation
- Last 15 min of each session — do not enter new trades

## When to Use

- Before generating trading signals (regime determines active combos)
- When user asks "should I trade now?" or "what's the market doing?"
- When evaluating why signals are being filtered out
- At morning open (9:00) to set initial trading posture
- At afternoon open (13:00) to re-assess regime after lunch break
- When ATR seems abnormally high or low

## Regime Classifications

| Regime | Conditions | Trading Posture |
|--------|-----------|-----------------|
| TRENDING | ADX > 25 AND abs(DI+ - DI-) > 10 | Trend combos only: A, D, E, F, J, M, O, X, W |
| RANGING | ADX < 20 | Mean-reversion combos only: G+, C, K, V, R |
| VOLATILE | ATR_5m > 1.5x ATR_SMA50 (or ATR > 4.2) | Reduce size, widen SL 1.5x, or skip entirely |
| NORMAL | None of the above | All combos active |

## Workflow

### Step 1: Fetch 15m Data

Load the latest 50 bars of 15m OHLCV data for VN30F1M using `src/data_fetcher.py`:

```python
from src.data_fetcher import DataFetcher
fetcher = DataFetcher()
df_15m = fetcher.get_ohlcv("VN30F1M", "15m", limit=50)
```

### Step 2: Compute Indicators

Calculate ADX(14), DI+(14), DI-(14), ATR(14), and ATR_SMA50:

```python
import pandas_ta as ta
df_15m.ta.adx(length=14, append=True)
df_15m.ta.atr(length=14, append=True)
atr_sma50 = df_15m['ATRr_14'].rolling(50).mean().iloc[-1]
```

### Step 3: Classify Regime

Apply the `detect_regime()` function from `scanner.py`:

```python
from scanner import detect_regime
regime = detect_regime(df_15m)
```

### Step 4: Report

Output the current regime with supporting metrics:
- ADX value and trend direction
- DI+ vs DI- spread
- Current ATR vs ATR_SMA50 ratio
- Active combo categories for this regime
- Confidence level (how clearly the regime is defined)

## Integration with Scanner

The regime detection runs automatically at the start of each scanner loop (every 35 seconds) in `scanner.py`. The `run_scan()` function filters combos based on regime before checking signals.

## Output Format

```
REGIME: TRENDING (ADX=32.4, DI+=28.1, DI-=14.3)
ATR: 3.8 pts (0.95x SMA50 → normal volatility)
Active combos: A, D, E, F, J, M, O, X, W (trend-following)
Disabled: G+, C, K, V, R (mean-reversion — wrong regime)
Confidence: HIGH (ADX well above threshold, clear DI separation)
```

## Intraday Time Windows

| Window | Time | Regime Behavior |
|--------|------|-----------------|
| AM Open | 9:00–9:15 | Volatile, regime not yet established. WAIT. |
| AM Prime | 9:15–10:45 | Best signal quality. Regime most reliable. |
| AM Wind-down | 10:45–11:15 | Trend may weaken. Reduce new entries. |
| AM Close | 11:15–11:30 | No new entries. Flatten if needed before lunch. |
| PM Open | 13:00–13:15 | Re-detect regime. May differ from morning. |
| PM Prime | 13:15–14:00 | Good signals but lower volume than AM. |
| PM Wind-down | 14:00–14:15 | Tighten SL. No new entries after 14:15. |
| PM Close | 14:15–14:28 | FLATTEN ALL. Force close by 14:28. |

## Key Principles

1. **15m timeframe for regime** — 5m is too noisy, 1h is too slow for intraday
2. **ATR > 4.2 is danger zone** — cost becomes too large relative to expected move
3. **Regime can change intraday** — re-evaluate every scanner cycle
4. **VOLATILE overrides all** — if ATR extreme, skip trading regardless of ADX
5. **Re-detect at PM open** — lunch break can reset market dynamics entirely
6. **No trades outside sessions** — 11:30-13:00 market is closed
7. **Flatten by 14:28** — absolute deadline, no exceptions

## Resources

- `scanner.py:detect_regime()` — Implementation of regime detection
- `strategy_config.yaml` — Combo-to-regime mapping
- `references/regime_rules.md` — Detailed regime transition rules
