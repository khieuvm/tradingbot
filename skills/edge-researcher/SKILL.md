---
name: edge-researcher
description: Research and discover new intraday trading edges for VN30F1M futures (sessions 9:00-11:30, 13:00-14:30 UTC+7) from real market data. Analyze compression zones, MFE patterns, adaptive exits, time-of-day effects, session-specific behavior. All edges must work within single session — no overnight holds, positions close by 14:28. Prioritizes data-driven iterative discovery.
---

# VN30F1M Edge Researcher

Data-driven, iterative research pipeline for VN30F1M intraday edges. The working style is: observe anomaly → write a focused research script → run it → read the numbers → form hypothesis → test → either kill or advance.

## Intraday Constraints (Non-negotiable)

1. **Entry → Exit within session** — no holding through lunch (11:30) or overnight
2. **Max holding time:** ~2 hours AM (prime: 9:15–11:15) or ~75 min PM (13:00–14:15)
3. **Force close at 14:28** — any strategy must complete before this
4. **Cost: 0.96 pts/trade** (slippage 0.5 + commission 0.46 round-trip)
5. **Sessions are structurally different** — PM is consistently stronger than AM for CB setups

## Session Characteristics

| Session | Characteristics | Notes |
|---------|----------------|-------|
| AM Open 9:00-9:15 | High vol, gap fills, ORB forming | Skip signals here |
| AM Prime 9:15-10:45 | Trend establishment, highest volume | Best CB window |
| AM Late 10:45-11:30 | Volume declining | No new entries |
| PM Open 13:00-13:15 | Gap from lunch | Skip |
| PM Prime 13:15-14:15 | Lower vol, compression common | CB works well here |
| PM Close 14:00-14:28 | Position flattening | Close only, no new entries |

**PM structural edge:** PM 13:45–14:15 has WR 75.9%, PF 8.73 on CB 5m. Traders closing EOD positions create one-directional flow when a compression breaks.

## When to Use

- User wants to improve an existing edge (exit, filter, adaptive logic)
- Investigating a specific pattern: MFE distribution, time windows, regime interaction
- After noticing live trading anomaly (e.g. "trades reverse at 4–6 pts too often")
- Monthly edge review

## Core Philosophy

**"Measure the problem first. Don't add a filter until you've quantified what you're filtering."**

Research sequence:
1. Identify anomaly from live or backtest data (e.g. "too many reversals at 4-6 pts MFE")
2. Write a dedicated script to measure it objectively
3. Find features that correlate with the problem
4. Test candidate filters/rules — measure impact on PnL, WR, PF
5. Only advance rules that improve P/D without killing frequency
6. Document findings in proven_edges.md regardless of outcome

## Research Scripts (Reference)

| Script | Purpose |
|--------|---------|
| `research_mfe46.py` | Analyze MFE 4-6 pts reversal zone: features that predict fail vs continue |
| `research_adaptive_exit.py` | Test adaptive exit at 4pts based on pre_move/ATR threshold |
| `bt_adaptive_3tf.py` | Backtest adaptive exit on 1m/3m/5m |
| `bt_trail_sweep.py` | Trail activation sweep (3/4/5 pts) + AM/PM time window analysis |
| `debug_3m.py` / `debug_1m.py` | CB deep analysis on 3m/1m with SL breakdown |

## Proven Edges Summary

| Edge | TF | WR | PF | P/D | Status |
|------|----|----|----|-----|--------|
| CB 3-bar compression | 5m | 65.0% | 4.54 | +6.05 | **Production** |
| CB + adaptive exit | 5m | 67.7% | 4.87 | +6.34 | **Apply now** |
| CB 5-bar compression | 3m | 41.7% | 1.02 | +0.03 | Monitoring (24d only) |
| CB 10-bar compression | 1m | 57.9% | 2.13 | +0.88 | Monitoring (24d only) |

See `references/proven_edges.md` for full detail.

## Research Methodology

### Step 1: Define the specific question

Frame as a measurable hypothesis with a testable outcome:
- "What % of trades that reach 4 pts MFE continue to 6+ pts?" ✓
- "Does pre-signal momentum predict MFE continuation?" ✓
- NOT: "Can I improve the strategy?" (too vague)

### Step 2: Load appropriate data

```python
from src.data_fetcher import DataFetcher
fetcher = DataFetcher()
df = fetcher.get_futures_ohlcv("VN30F1M", start, end, interval="5m")
# API limits: 5m ~144d, 3m ~28d, 1m ~28d
# Holiday fix: filter df[df['date'] >= date(2026, 5, 2)] for 3m/1m
```

### Step 3: Establish baseline before adding any filter

Always run baseline first, then test changes:
```
BASELINE:    223 trades | WR 65.0% | PF 4.54 | +6.05/d
With filter: 223 trades | WR 67.7% | PF 4.87 | +6.34/d (+0.29/d)
```

Show the delta explicitly. If PF improves but P/D drops, that's a frequency trade-off worth flagging.

### Step 4: Feature engineering for MFE analysis

Key features to capture at signal bar:
```python
# Pre-signal momentum (3 bars before, in trade direction)
pre_bars = df.iloc[max(0, i_pos-3):i_pos]
pre_move = (pre_bars.iloc[-1]['close'] - pre_bars.iloc[0]['open']) * direction
pre_ratio = pre_move / atr  # normalize to ATR

# Signal bar quality
sig_range_atr = row['range'] / atr
sig_body_ratio = abs(row['close'] - row['open']) / row['range']

# Volume
vol_ratio = row['volume'] / df['volume'].rolling(20).mean()

# Speed to MFE level
bars_to_4 = None  # set when mfe crosses 4.0 in simulation loop
```

### Step 5: Adaptive logic at MFE checkpoints

Test exit decisions AT specific MFE levels, not just entry filters:
```python
# At 4pts MFE — check if this trade should exit or continue
if mfe >= 4.0 and session == 'AM' and pre_ratio > 0.8:
    exit_p = b['close']  # take 4pts, don't trail
    exit_r = 'ADAPT_EXIT'
    break
```

This is more powerful than entry filters because you use bar-by-bar context.

### Step 6: Multi-TF validation

After finding a rule on 5m, always test same rule (normalized to ATR) on 3m/1m:
- 5m: pre_ratio > 0.8 → threshold in 5m ATR terms (~3 pts with ATR 3.5)
- 3m: same ratio but lower threshold may work (try 0.6) because 3m moves are smaller
- 1m: check if enough adapt signals fire (may be negligible)

### Step 7: Document regardless of outcome

Both positive AND negative findings go to `references/proven_edges.md` or `references/anti_patterns.md`. A confirmed anti-pattern saves future research time.

## Output Format

```
RESEARCH: [specific question]
Data: [TF, days, trades]
═══════════════════════════════════════════

Problem measured:
  Baseline: 223 trades | WR 65.0% | PF 4.54 | +6.05/d
  MFE 4-6 zone: 48 trades, WR 50.0% (reversal zone identified)

Feature correlations (TIEP_TUC >=6 vs DUNG_LAI 4-6):
  Pre-move (3 bars): 0.04 vs 0.86 (**most discriminating**)
  Vol ratio:         0.95 vs 1.09 (**)
  EMA alignment:     0.55 vs 0.54 (no difference)

Candidate rules tested:
  AM pre_ratio > 0.8 → exit@4: +6.34/d (+0.29 vs baseline) ✓
  Vol >= 1.2x filter:           +1.87/d (kills frequency) ✗

Verdict: Apply adaptive exit rule. Vol filter is confidence boost only.
Implementation: combos/<name>.py inheriting BaseCombo
```

## Key Research Findings (Running Log)

**2026-06 — MFE 4-6 Reversal Zone:**
- 30% of trades that reach 4 pts MFE reverse before 6 pts
- Root cause: `pre_move/ATR > 0.8` at signal = exhaustion, not fresh breakout
- Fix: `if AM and pre_ratio > 0.8 → exit@4pts` — improves +0.29/d on 5m
- 3m: same rule at ratio > 0.6 improves from breakeven to +0.52/d (24d sample)
- 1m: only 1 adapt signal in 24d — negligible

**PM session structural superiority:**
- PM 13:45–14:15 consistently WR 75%+, PF 8+ across all TFs
- PM needs no filtering — natural edge from EOD position closing flow

**EMA9/EMA21 alignment — CONFIRMED USELESS for CB:**
- Aligned vs not-aligned: WR 64.7% vs 65.4% — no difference
- Do not use as entry filter

## Resources

- `references/proven_edges.md` — All validated findings with stats
- `references/anti_patterns.md` — Confirmed dead ends
- `backtest/engine.py` — CB backtest engine (trail sweep, simulation)
- `combos/base.py` — BaseCombo interface for new strategies
- `combos/cb.py` — CB reference implementation
- `src/data_fetcher.py` — Data access (note API limits above)
- `research/` — Place new analysis scripts here
