# VN30F1M Proven Edges

## 1. Session Opening Range Breakout (ORB)

**Setup:** First 2 bars (10 min) of session establish range. Trade breakout of that range.

**Parameters:**
- Bar count: 2 (first 10 minutes of session)
- Max range: 6 pts (wider = higher SL = skip)
- Entry: break of high/low of range + 0.5 pts buffer
- SL: opposite side of range
- Exit: trailing at 2.0x ATR after +5 pts profit

**Statistics (240-day sample):**
- Frequency: 0.5-0.7 signals/day
- WR: ~38% (low but compensated by trailing)
- PF: 1.15-1.25
- Works best: LOW_VOL and NORMAL ATR regimes
- Fails in: HIGH_VOL (ATR > 4.2)

**Why it works:** Session open concentrates order flow. Range establishes supply/demand levels. Breakout captures the directional move from trapped traders.

---

## 2. CB — Compression Breakout (5m)

**Setup:** 3 consecutive 5m bars with range < 0.7x ATR(14). Trade breakout of the compression zone.

**Entry signal:** 3 bars max_range < 0.7x ATR(14)

**Entry filters:**
- ATR(14): 2.5–4.5 pts (skip whipsaw > 4.5, skip micro < 2.5)
- RSI(14) < 70 (skip overbought exhaustion — no lower bound)
- Dedup: min 5 bars between signals
- Time windows: AM 9:15–10:45, PM 13:15–14:15

**Direction:** From NEXT bar close vs signal bar close (not current bar). This is critical — do NOT use signal bar direction.

**Risk/Exit (AM session):**
- SL: 1.2x ATR
- Breakeven: move SL to entry at +4 pts MFE when ATR ≥ 3.5
- Trail activation: +5 pts MFE
- Trail distance: 2.0x ATR
- Max hold: 24 bars (2 hours)

**Risk/Exit (PM session):**
- SL: 1.0x ATR (tighter, less time)
- Trail activation: +4 pts MFE
- Trail distance: 1.5x ATR
- Max hold: 12 bars (1 hour)

**Adaptive exit — AM "exhaustion" filter:**
- Compute: `pre_move = (close[-1] - open[-3]) * direction` (3 bars before signal)
- `pre_ratio = pre_move / ATR`
- If `session == AM` and `pre_ratio > 0.8`: exit immediately at +4 pts MFE instead of trailing
- Rationale: pre_move > 0.8x ATR = prior momentum exhausted; trade likely reverses 4–6 pts zone
- **Result: +6.34/d vs +6.05/d baseline (+0.29/d improvement, 21 extra exits on 129 days)**

**Statistics — Full backtest (129 days, cost 0.96/trade):**
- 223 trades | WR 65.0% | PF 4.54 | +780.9 pts | +6.05/d
- With adaptive exit: WR 67.7% | PF 4.87 | +818.2 pts | **+6.34/d**
- AM: 142 trades | WR 62.7% | PF 3.80 | +418.7 pts
- PM: 81 trades | WR 70.4% | PF 6.23 | +364.8 pts

**Time window breakdown (with adaptive exit):**
| Window | T | WR | PF | PnL | avgMFE |
|--------|---|----|----|-----|--------|
| AM 09:15-10:00 | 67 | 59.7% | 3.93 | +250.8 | 8.1 |
| AM 10:00-10:45 | 69 | 71.0% | 4.04 | +170.9 | 6.6 |
| PM 13:15-13:45 | 50 | 68.0% | 6.09 | +245.3 | 8.6 |
| **PM 13:45-14:15** | 29 | **75.9%** | **8.73** | +123.3 | 7.7 |

**MFE zone analysis (key risk zone: 4–6 pts):**
- 48 trades reach 4–6 pts MFE: WR only 50.0% → reversal zone
- Pre_ratio > 0.8 (exhaustion signal): 40% fail@4-6 → exit early at 4 pts
- Vol > 1.2x avg: fail@4-6 only 18%, but only 47 trades/129d
- EMA9/EMA21 alignment: NO predictive value in 4-6 zone (54% vs 55%)

**OOS validation (30 days out-of-sample, cost 0.96):**
- 30 trades | WR 50.0% | PF 1.72 | +1.50/d
- Confirms genuine edge (not curve-fit)

**Note on old stats (cost 1.74):** Old: WR 60.8%, PF 3.23, +4.85/d. Corrected cost 0.96 improves PF from 3.23 → 4.54.

**Volume insight (not applied as hard filter):**
- Vol >= 1.2x avg: WR 76.6%, PF 11.79 (47 trades) — confidence boost only
- Vol < 0.8x avg: WR 60.7%, PF 3.81 (140 trades) — still tradeable
- Hard vol filter kills trade frequency too much (47→47 vs 223 baseline)

**Entry improvements that DON'T work:**
- Confirmation candle (wait 1-2 bars): WR drops to 40% — kills edge
- Limit order (0.2-0.5x ATR offset): fill rate < 8% — impractical
- EMA9/EMA21 alignment filter: no improvement over baseline
- RSI lower bound filter: reduces trade count with no quality gain

**Why it works:** Compression = supply/demand equilibrium. When 3 bars squeeze below 0.7x ATR, both sides are in balance. Breakout resolves the equilibrium → trapped side covers → directional impulse. Edge is in predicting VOLATILITY EXPANSION, not direction. Trailing stop captures the move regardless of direction.

---

## 3. CB — Compression Breakout (3m) — Early Stage

**Setup:** 5 consecutive 3m bars with range < 0.7x ATR(14). Equivalent time to 3-bar 5m (15 min compression).

**Entry filters:** ATR 1.5–3.0 pts, RSI < 70, dedup 8 bars, same time windows as 5m.

**Risk/Exit:** Same structure as 5m but scaled:
- AM: SL 1.2x ATR, trail@5pts/2.0x, max_hold 40 bars
- PM: SL 1.0x ATR, trail@4pts/1.5x, max_hold 20 bars

**Statistics (24 days from 2026-05-02, cost 0.96):**
- Baseline: 24 trades | WR 41.7% | PF 1.02 | breakeven
- With adaptive exit (pre_ratio > 0.6): WR 54.2% | PF 1.35 | +0.52/d
- AM: losing (PF 0.92 baseline); PM: PF 2.27 with only 3 trades

**Time window note:** AM 9:15-10:00 (WR 61.5%, PF 2.05) >> AM 10:00-10:45 (WR 37.5%, losing)

**Warning:** 24-day sample is insufficient for statistical confidence. Need 60+ days minimum.
**API limitation:** KBS API returns max ~28 days for 3m data. Wait for more data before deploying.

**3m adaptive exit:** pre_ratio > 0.6 threshold (lower than 5m because 3m ATR is smaller) — 6 exits, dramatic improvement but tiny sample.

---

## 4. CB — Compression Breakout (1m) — Pilot Only

**Setup:** 10 consecutive 1m bars with range < 0.7x ATR(14). 10 bars = same 10 min as 3-bar 5m.

**Entry filters:** ATR 0.7–1.5 pts, RSI < 70, dedup 15 bars.

**Risk/Exit:** Same structure:
- AM: SL 1.2x ATR, trail@5pts/2.0x, max_hold 90 bars
- PM: SL 1.0x ATR, trail@4pts/1.5x, max_hold 45 bars

**Statistics (24 days from 2026-05-02, cost 0.96):**
- 19 trades | WR 57.9% | PF 2.13 | +0.88/d
- AM: 15 trades, WR 46.7%, PF 1.29 (weak)
- PM: 4 trades, WR 100%, PF 999 (too few to assess)
- Adaptive exit: only 1 exit triggered — negligible

**Warning:** 19 trades / 24 days = too small for any conclusion. Monitor only.

**TF comparison summary:**

| TF | Days | Trades | WR | PF | P/D | Status |
|----|------|--------|----|----|-----|--------|
| 5m | 129 | 223 | 65.0% | 4.54 | +6.05 | **Production** |
| 3m | 24 | 24 | 41.7% | 1.02 | +0.03 | Monitoring |
| 1m | 24 | 19 | 57.9% | 2.13 | +0.88 | Monitoring |

---

## 5. VN Market Constraints

**One-direction rule:** VN derivatives — only ONE direction at any time (no simultaneous long+short on same contract).
- Portfolio direction lock is mandatory
- Flip requires closing ALL positions first

---

## 6. Cost & Breakeven Rules

**Contract:** 1 point = 100,000 VND. Margin ~36,000,000 VND/contract.

**Round-trip cost: 0.96 pts** (slippage 0.5 + commission 0.46)
- Tax + broker: 46,000 VND = 0.46 pts (total open+close)
- Slippage: 0.5 pts

**ATR filter validated (180-day backtest):**
- ATR > 4.5: SL rate 44%, PF drops to 1.62 → skip
- ATR < 2.5: WR 41.7%, moves too small → skip
- ATR 2.5–4.5: sweet spot, WR 62.7%, PF 3.07

---

## 7. Data Limitations (API)

**KBS API data availability:**
- 5m: ~144 trading days (6 months)
- 3m: ~28 trading days (1 month)
- 1m: ~28 trading days (1 month)

**Holiday duplicate data bug:** April 30 (Liberation Day) and May 1 (Labor Day) return cloned data of April 29 (all open=2029.5). Always filter from day after holidays: `df[df['date'] >= date(2026, 5, 2)]`.

---

## Key Insight

Direction prediction is nearly impossible on VN30F1M 5m (62% of bars lead to 8+ pt moves, but direction is 50/50). The edge is in predicting VOLATILITY EXPANSION from compression zones, then using trailing stops to capture the move regardless of direction.

**PM session structural advantage:** PM 13:45–14:15 consistently outperforms AM across all TFs. PM traders are closing positions before EOD — one-directional flow is more common.
