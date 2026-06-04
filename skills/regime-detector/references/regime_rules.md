# VN30F1M Regime Detection Rules

## Regime Definitions

### TRENDING
- **Primary:** ADX(14) on 15m > 25
- **Secondary:** abs(DI+ - DI-) > 10
- **Both conditions required**
- **Confidence:** HIGH if ADX > 30, MEDIUM if ADX 25-30

### RANGING
- **Primary:** ADX(14) on 15m < 20
- **No secondary required**
- **Confidence:** HIGH if ADX < 15, MEDIUM if ADX 15-20

### VOLATILE
- **Primary:** 5m ATR(14) > 1.5x rolling ATR SMA(50)
- **OR:** 5m ATR > 4.2 pts (absolute threshold)
- **Overrides TRENDING/RANGING** — if volatile, always volatile
- **Confidence:** HIGH if ATR > 2x SMA50

### NORMAL
- **Default:** when none of the above apply
- **ADX between 20-25 OR DI spread < 10**
- **ATR within normal range**

## Transition Rules

- Regime must persist for 3 consecutive 15m bars to be confirmed
- Single-bar regime change is noise → keep previous regime
- VOLATILE can trigger immediately (no persistence needed) — safety first

## Combo-Regime Mapping

### TRENDING combos (trade with trend)
A, D, E, F, J, M, O, X, W

### RANGING combos (mean-reversion)
G+, C, K, V, R

### ALL-regime combos (rare, must be robust)
Only combos that pass backtest-validator in both regimes

## ATR Danger Zones

| ATR Range | Classification | Action |
|-----------|---------------|--------|
| < 2.5 | TOO_LOW | Skip — moves won't cover cost |
| 2.5-3.0 | LOW | Reduce TP target, tighter trailing |
| 3.0-4.0 | OPTIMAL | Normal trading parameters |
| 4.0-4.2 | ELEVATED | Monitor closely, normal params |
| > 4.2 | DANGER | Reduce size 50% or skip entirely |
| > 5.0 | EXTREME | Do not trade |

## Time-Based Regime Adjustments

### Morning Session (9:00-11:30)
- **9:00-9:15:** Regime unreliable, opening volatility spike. DO NOT TRADE.
- **9:15-9:30:** Regime forming. Can enter if regime signal is strong (ADX>28 or ADX<17).
- **9:30-10:45:** Prime time. Regime most reliable. Full confidence.
- **10:45-11:15:** Trend may weaken approaching lunch. Reduce entry confidence.
- **11:15-11:30:** NO new entries. Close positions not on trailing.

### Lunch Break (11:30-13:00)
- Market is CLOSED. No data, no signals.
- Regime at 11:30 does NOT carry to 13:00.
- Always re-detect regime fresh at PM open.

### Afternoon Session (13:00-14:30)
- **13:00-13:15:** Re-detect regime. Afternoon dynamics often differ from morning.
- **13:15-14:00:** PM prime time. Often more RANGING than AM (lower volume).
- **14:00-14:15:** Last entries allowed. Only HIGH confidence regime signals.
- **14:15-14:28:** NO new entries. Flatten all positions by 14:28. No exceptions.

### Session Regime Differences
- AM tends toward TRENDING (institutional order flow at open)
- PM tends toward RANGING (lower volume, profit-taking)
- If AM was VOLATILE, PM often calms down — don't assume continuity
- Re-check ADX/ATR at 13:00 regardless of AM classification
