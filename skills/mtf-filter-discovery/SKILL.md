---
name: mtf-filter-discovery
description: Multi-timeframe indicator discovery for VN30F1M signal filtering. Compute HTF indicators (5m for 1m signals, 15m for 5m signals), compare winner/loser distributions, find discriminating thresholds. Simple rules that outperform ML with less overfitting and more trades kept.
---

# VN30F1M MTF Filter Discovery

At each trading signal on the lower timeframe (LTF), look at the most recently completed higher timeframe (HTF) bar's indicators. Compare indicator distributions between winners and losers. Find the single indicator + threshold that best separates them.

## Why MTF Filters Beat ML

- **Simpler** — 1 rule vs 50 features. Less overfitting.
- **More trades kept** — typically 30-60% pass rate vs ML's 10-20%
- **Interpretable** — "take BUY when 5m EMA slope is up" makes logical sense
- **Robust** — fewer degrees of freedom means less curve-fitting

## When to Use

- After finding a profitable LTF combo that has too many losers (WR < 55%)
- When ML filter rejects too many winners (high false rejection rate)
- Exploring whether HTF alignment can rescue a "mediocre" combo
- Monthly edge review — re-check if filter thresholds have shifted

## Timeframe Ratios

| LTF (signals) | HTF (filter) | Ratio | Function |
|--------------|-------------|-------|----------|
| 1m | 5m | 5:1 | `compute_5m_from_1m()` |
| 5m | 15m | 3:1 | `compute_15m_bars()` |

## Decision Gates

### GATE 1: Sample Size
- LTF combo has >= 80 labeled signals (winners + losers)?
- **NO** -> Insufficient for reliable indicator comparison. Extend data period or merge sessions.
- **YES** -> Proceed to indicator analysis.

### GATE 2: Discrimination Strength
- Best indicator has Cohen's d > 0.20 between winners and losers?
- **NO** -> No single HTF indicator discriminates well. Try: (a) ML ensemble via `strategy_filter.py`, (b) combined 2-indicator filter, (c) accept that HTF doesn't help this combo.
- **YES** -> Test threshold sweep.

### GATE 3: Keep Ratio
- Optimal threshold retains >= 30% of all signals?
- **NO** -> Filter is too aggressive. Either: widen threshold (accept lower WR improvement), or try a different indicator with broader discrimination.
- **YES** -> Check profitability.

### GATE 4: Profitability Improvement
- Filtered PF > baseline PF + 0.3?
- **NO** -> Filter adds complexity without meaningful improvement. Use unfiltered.
- **YES** -> Deploy. Document in proven_edges.md.

## Methodology

### Indicator Zoo (31 HTF indicators computed)

**Trend:** ema8, ema21, ema50, ema8_slope, ema21_slope, ema_align (-3 to +3)
**Momentum:** rsi_14, rsi_7, macd_hist, macd_line, macd_signal, macd_hist_dir, mom_3, mom_5
**Volatility:** adx, di_plus, di_minus, di_spread, atr, bb_pos, kc_pos
**Structure:** stoch_k, stoch_d, hh (higher highs), ll (lower lows), body_pct, range
**Volume:** vol_ratio (vs 10-bar SMA)
**Distance:** dist_high_5 (from 5-bar high), dist_low_5, bars_above_ema21

### Threshold Sweep Algorithm

```python
for percentile in range(10, 91, 5):  # 17 thresholds
    thresh = np.percentile(all_values, percentile)
    # Test "above thresh" and "below thresh" as filter directions
    # Score = (WR_improvement * 100) + (keep_ratio * 20)
    # Require: keep_ratio >= 0.30, filtered WR > baseline WR
```

### Scoring Formula

```
score = (filtered_WR - baseline_WR) * 100 + keep_ratio * 20
```

This balances WR improvement against trade frequency preservation.

## CLI Commands

```bash
# 5m strategies filtered by 15m indicators
python -m ml.mtf_indicator_discovery --tf 5m

# 1m strategies filtered by 5m indicators
python -m ml.mtf_1m_with_5m

# All 54 combos on 1m with automatic best 5m filter selection
python -m ml.scan_all_1m_combos
```

## Key Functions

| Function | File | Purpose |
|----------|------|---------|
| `compute_5m_from_1m(df_1m)` | ml/mtf_1m_with_5m.py | Aggregate 1m→5m bars |
| `compute_15m_bars(df_5m)` | ml/mtf_indicator_discovery.py | Aggregate 5m→15m bars |
| `compute_5m_indicators(df_5m)` | ml/mtf_1m_with_5m.py | 31 indicators on 5m |
| `compute_htf_indicators(df_15m)` | ml/mtf_indicator_discovery.py | 32 indicators on 15m |
| `map_1m_to_5m(df_1m, df_5m)` | ml/mtf_1m_with_5m.py | Map each 1m bar to completed 5m bar |
| `map_5m_to_15m(df_5m, df_15m)` | ml/mtf_indicator_discovery.py | Map each 5m bar to completed 15m bar |
| `analyze_indicator_discrimination(w, l, name)` | ml/mtf_indicator_discovery.py | Cohen's d + threshold sweep |
| `test_combined_filters(signals, cols, top, n)` | ml/mtf_indicator_discovery.py | Pair combinations test |
| `find_best_5m_filter(signals, df_5m, mapping, cols, n)` | ml/scan_all_1m_combos.py | Auto-select best filter |

## Proven Findings

| LTF Combo | HTF Filter | Effect | Keep% |
|-----------|-----------|--------|-------|
| momentum_trend 1m PM/SELL | ema8_5m > 1886 | WR 55%→68% | 55% |
| momentum_trend 1m PM/BUY | ema21_slope_5m > 0 | WR 47%→58% | 50% |
| macd_cross 1m AM/SELL | ll_5m >= 1 | WR 47%→71% | 33% |
| macd_cross 1m PM/BUY | ema21_slope_5m > 0 | WR 49%→78% | 35% |
| fibonacci 1m PM/SELL | macd_hist_5m > 0.02 | WR 38%→53% | 40% |
| heikin_ashi 1m AM/SELL | macd_line_5m < 5.95 | WR 50%→56% | 80% |
| market_structure 5m ALL/BUY | bb_pos_15m > 0.46 | WR→61% | 60% |
| sr_horizontal 5m AM/SELL | di_plus_15m < 24.8 | WR→67% | 45% |

## Output Format

```
INDICATOR          | Cohen's d | Threshold | Direction | WR    | Keep% | PnL
-------------------+-----------+-----------+-----------+-------+-------+------
ema8_5m            |    +0.35  |   1886.0  |   above   |  68%  |  55%  | +416
ema21_slope_5m     |    +0.28  |    0.000  |   above   |  58%  |  50%  | +347
ll_5m              |    +0.42  |    1.000  |   above   |  71%  |  33%  | +121
```

## Key Principles

1. **No look-ahead** — always use the COMPLETED HTF bar (target_group - 1), never the current forming bar
2. **Keep ratio >= 30%** — a filter that rejects 80% of trades is useless even with high WR
3. **Simple beats complex** — one indicator threshold beats multi-indicator ML for most combos
4. **Direction-aware** — BUY signals need different HTF conditions than SELL signals
5. **Re-validate monthly** — absolute thresholds (like ema8_5m > 1886) drift with price; relative indicators (slope > 0) are more stable

## Resources

- `ml/mtf_indicator_discovery.py` — Main discovery script for 5m→15m
- `ml/mtf_1m_with_5m.py` — Discovery script for 1m→5m
- `ml/scan_all_1m_combos.py` — All-combos scanner with auto-filter
- `ml/reports/` — Output reports (JSON)
- `skills/strategy-optimizer/SKILL.md` — Parent optimization workflow
