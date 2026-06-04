# VN30F1M Anti-Patterns (Proven NOT to Work)

Things tested and confirmed to have no edge, or confirmed to hurt. Don't re-test without a specific new hypothesis.

## 1. Bollinger Band Squeeze → Breakout Direction

**Tested:** BB squeeze (bandwidth < 20th percentile) → direction prediction.
**Result:** Squeeze is ANTI-correlated with moves. Squeeze → continued sideways more often than breakout.
**Why:** BB squeeze detects low volatility but doesn't predict when volatility will return or which way.

---

## 2. Single Indicator Direction Prediction

**Tested:** RSI, MACD, Stochastic, CCI, Williams %R — all single indicators.
**Result:** Max WR achievable: 41%. Not enough to overcome 0.96 pts cost.
**Why:** VN30F1M 5m is near-random walk for direction. No single indicator predicts 50/50 direction.

---

## 3. EMA Crossovers (Any Combination)

**Tested:** EMA 9/21, 5/20, 10/50 crossovers on 5m.
**Result:** Too slow for intraday. By crossover confirmation, move is 60–70% done.
**Why:** EMAs are lagging. Intraday moves are faster than EMA lag.

---

## 4. Volume-Only Signals

**Tested:** Volume spikes (> 2x avg) as entry trigger.
**Result:** Volume spikes happen at both tops and bottoms equally. No directional edge.
**Why:** High volume = high activity = could be buying OR selling.

---

## 5. Fixed TP Without Trailing

**Tested:** Fixed TP of 8–10 pts.
**Result:** PF drops 15–20% vs trailing. Winners are capped while losers run full SL.
**Why:** 15–20 pt moves happen frequently enough that capping at 8–10 leaves significant money on the table.

---

## 6. HIGH_VOL Regime Trading (ATR > 4.5)

**Tested:** Same CB setups during ATR > 4.5 periods.
**Result:** SL hit rate 44%, PF drops to 1.62.
**Why:** Wider candles = more noise inside compression → SL gets clipped more often. Cost unchanged.

---

## 7. Strong Bar Continuation

**Tested:** After large bullish bar (body > 1.5x ATR), buy continuation.
**Result:** No edge. Strong bars equally likely to continue or reverse.
**Why:** Strong bars often represent END of move (exhaustion), not beginning.

---

## 8. Mean Reversion on Trending Days

**Tested:** Fade-the-move when ADX > 30.
**Result:** Consistent losses. Trends extend further than expected.
**Why:** Trending = one-sided order flow. Fading gets destroyed by continuation.

---

## 9. EMA9/EMA21 Alignment as CB Entry Filter

**Tested:** Only take CB signals when EMA9 is aligned with trade direction vs EMA21.
**Result:** Aligned: WR 64.7%, PF 5.12, 119 trades. Not aligned: WR 65.4%, PF 3.91, 104 trades.
**NO improvement in direction.** Both groups have similar WR. Aligned PF is slightly better but this is due to different MFE distribution, not signal quality.
**Why:** CB edge is about volatility expansion prediction, not direction. EMA alignment is about trend — irrelevant for compression breakouts.
**Critical finding:** In the MFE 4–6 reversal zone, EMA alignment = 55% for fail vs 54% for continue — statistically identical. EMA tells you nothing about whether a trade will continue past 4 pts.

---

## 10. Confirmation Candle (Wait 1–2 Bars After Signal)

**Tested:** Wait 1–2 bars after CB signal fires before entering.
**Result:** WR drops from 65% to ~40%.
**Why:** The edge is in predicting the breakout BEFORE it happens (at compression point). Waiting for confirmation means entering AFTER the breakout move has already started — you miss the early move and enter during pullback.

---

## 11. MI (Momentum Impulse) Strategy

**Tested:** MI strategy on 1m/3m/5m VN30F1M.
**Result:** WR 7–22%, PF 0.01–0.50 across all timeframes. Definitively unprofitable.
**Why:** VN30F1M intraday momentum is too noisy. Impulse moves reverse too frequently within session.

---

## 12. Vol >= 1.0-1.2x as Hard Entry Filter (CB)

**Tested:** Only take CB signals when volume >= 1.0x or 1.2x rolling average.
**Result:** Vol >= 1.0x: 65 trades, PF 8.12, +2.39/d. Vol >= 1.2x: 47 trades, PF 11.79, +1.87/d.
**Better PF per trade — but total P/D is LOWER than baseline (+6.05/d).**
**Why it doesn't work as hard filter:** Low-volume bars (< 1.0x avg) still produce winning trades. Vol filter removes 156 trades/129 days, killing 3.66 pts/day of P/D that was still profitable. The quality gain doesn't compensate the frequency loss.
**Use as:** Confidence boost or position sizing signal. If vol >= 1.2x, be more aggressive on size.

---

## 13. Pre-Signal Strong Momentum as Continuation Signal

**Tested:** If pre_move (3 bars before signal) > 3 pts in trade direction → expect large continuation.
**Result:** Strong pre_move (> 3 pts) group has the HIGHEST fail@4-6 rate: 40% vs 27% baseline.
**Counterintuitive:** More prior momentum → MORE likely to reverse after 4 pts.
**Why:** Strong momentum before a compression breakout = exhaustion play, not fresh breakout. Price already moved → less fuel for continuation. The cleanest CB breakouts come from flat/neutral prior bars.
**Actionable:** If AM session and pre_move/ATR > 0.8 → flag as "exhaustion breakout" → exit at 4 pts instead of trailing (tested: +0.29/d improvement on 5m).

---

## 14. ATR < 2.5 (Low Volatility Regime) for CB

**Tested:** CB signals when ATR < 2.5.
**Result:** WR 41.7%, moves too small after cost.
**Why:** With ATR < 2.5, the compression zone is tiny (< 1.75 pts max range). SL = 1.2x ATR = 3 pts. Expected move isn't large enough to justify 0.96 pts cost + 3 pt SL.
