# Education Combos Research — Intraday Strategies for VN30F1M

**Date compiled:** 2026-06-05
**Sources:** Quantified Strategies, Trading Strategy Guides, Investopedia, Larry Connors
  research, John Carter TTM Squeeze docs, Alexander Elder Trading for a Living,
  futures.io community notes, TradingView published strategy stats
**Purpose:** Identify programmable intraday strategies with specific rules, evaluate
  applicability to VN30F1M 5m, rank for backtesting priority.

---

## Anti-pattern reminder (from references/anti_patterns.md)

Before each entry, confirmed anti-patterns are flagged. Skip anything already disproven:
- EMA crossovers (too slow)
- Single indicator direction (max 41% WR)
- BB squeeze for direction (anti-correlated on VN30F)
- Strong bar continuation (exhaustion signal)
- Confirmation candle wait (kills CB edge)

---

## Strategy 1: NR7 Volatility Breakout

**Source:** Larry Connors & Cesar Alvarez, "Short-Term Trading Strategies That Work" (2008);
  quantifiedstrategies.com/nr7-trading-strategy/
**Applicability to VN30F1M:** HIGH

**Concept:**
The Narrow Range 7 (NR7) bar has the smallest high-low range of the preceding 7 bars.
This marks a volatility contraction point; range expansion typically follows. Unlike CB
which uses a 3-bar rolling window, NR7 uses a 7-bar lookback, making it a complementary
compression detector operating at a different frequency.

**Parameters:**
- Lookback window: 7 bars (the NR7 bar itself is bar 0, compare its range to bars -1 to -6)
- Range definition: bar_range = high - low
- Buy trigger: price crosses above NR7 bar high + small buffer (0.1 pts for VN30F)
- Sell trigger: price crosses below NR7 bar low - small buffer
- Stop loss: opposite side of NR7 bar (NR7 bar low - buffer for longs, high + buffer for shorts)
- Initial risk = NR7 bar full range (narrow by definition)
- Exit options: fixed ATR multiple (2x ATR from entry), or trail at 1x ATR after 2x MFE
- Also tested as mean-reversion (fade the breakout after 3 bars of failure) — but breakout
  direction is the primary edge

**Reported backtest results:**
- Connors equity index research: ~55% WR on futures, PF ~1.5-1.8 (before slippage)
- quantifiedstrategies.com stocks study: 60% WR on SPY daily; intraday untested
- After 0.96 pts cost: requires minimum signal-bar range of 0.5 pts to be viable
- Frequency: NR7 signals appear approximately 15-20% of all bars on 5m data

**Intraday fit:**
- Sessions 9:00-11:30, 13:00-14:30: yes, signal is instantaneous (no lag)
- Expected frequency: 2-4 signals/session on 5m (roughly 1 per hour)
- Compatible with trailing exit: yes, SL is compact due to narrow bar
- Works without direction prediction: yes — enter both sides simultaneously as OCA, or
  wait for whichever trigger fires first (direction emerges from price action)

**Key difference from CB:**
CB requires 3 consecutive narrow bars (streak compression). NR7 requires only 1 bar that
is narrower than all of the prior 6. CB = streak-based; NR7 = single-point extremum.
They can co-occur: an NR7 that is also the 3rd consecutive compression bar is a very high
confidence setup.

**Integration idea:**
Use NR7 as a secondary confirmation for CB: if the compression bar (bar before breakout)
is also an NR7 bar, classify the signal as CB+NR7 and increase position or raise confidence
score. Alternatively, test NR7-only signals (no 3-bar streak required) as a standalone
strategy with looser entry criteria.

**Next step:**
Write `research/nr7_breakout.py`:
- Load 5m VN30F1M data (144d)
- Compute bar range, rolling 7-bar min range
- Flag NR7 bars (range == rolling min)
- Simulate breakout entry on next bar with ATR-based SL
- Report WR, PF, frequency, avg trade duration
- Compare NR7 subset vs NR7 that overlaps with CB signal

---

## Strategy 2: TTM Squeeze (Bollinger Band Inside Keltner)

**Source:** John Carter, "Mastering the Trade" (2006); TradingView built-in indicator
  "TTM_Squeeze" by lazybear; quantifiedstrategies.com/ttm-squeeze-trading-strategy/
**Applicability to VN30F1M:** HIGH (as volatility predictor, NOT direction predictor)

**Concept:**
When Bollinger Bands (BB, period=20, sd=2.0) are entirely inside Keltner Channels
(KC, period=20, ATR multiplier=1.5), the market is in a "squeeze" — volatility is at an
extreme low. When BB expands back outside KC, the squeeze fires and a volatility expansion
follows. A momentum histogram (modified MACD style) indicates direction of the expansion.

**Parameters:**
- BB: SMA(20), stddev multiplier = 2.0
- KC: EMA(20), ATR(20) multiplier = 1.5 (original Carter); also tested at 1.0 for tighter
- Momentum: value = close - avg( avg(highest(high,20), lowest(low,20)), SMA(close,20) )
- Squeeze ON: BB_upper < KC_upper AND BB_lower > KC_lower
- Squeeze FIRE (entry signal): bar where squeeze turns OFF (BB expands outside KC)
- Direction: momentum histogram > 0 at fire bar → Long; < 0 at fire bar → Short
- Stop loss: low of lowest bar during squeeze period (for longs); high of highest bar (shorts)
- Exit: momentum histogram crosses zero (reversal) OR session end
- Alternative exit: trail at 1x ATR after 2x ATR MFE

**Reported backtest results:**
- Carter claims 70%+ WR on trending instruments; this is disputed by independent testers
- Independent TradingView community tests: ~54-58% WR on 5m equity index futures
- Quantified Strategies review: squeeze works as a VOLATILITY predictor (expansion follows
  squeeze in 68% of cases within 5 bars), but direction prediction from histogram is ~52%
  (essentially random)
- KEY WARNING: On VN30F1M backtests done in this project, BB squeeze for direction is in
  the anti-patterns list (anti-correlated). The VOLATILITY prediction aspect is valid;
  the direction histogram is not.

**Anti-pattern flag:**
"BB squeeze for direction" is listed as confirmed anti-pattern for VN30F1M. The squeeze
FIRE event itself (volatility expansion) is valid — identical to CB logic. The momentum
histogram direction component should NOT be used as a standalone direction signal.

**Intraday fit:**
- Sessions: yes
- Expected frequency: 1-2 squeeze fires/session (longer squeeze = rarer signal)
- Compatible with trailing exit: yes
- Works without direction prediction: YES — use squeeze fire as volatility trigger only,
  let price action (H+0.1 or L-0.1 trigger) determine direction

**Integration idea:**
Map TTM Squeeze fire events to CB signal timing. If a CB signal occurs within 2 bars of
a TTM Squeeze fire, this is a dual-confirmation high-confidence volatility expansion signal.
Test: does CB + TTM co-occurrence improve WR above CB alone?

**Next step:**
Write `research/ttm_squeeze_filter.py`:
- Compute BB(20,2) and KC(20,1.5) on 5m data
- Flag squeeze ON periods (BB inside KC)
- Flag squeeze FIRE bars (transition from squeeze ON to OFF)
- Cross-reference with existing CB signals
- Report: CB WR when TTM squeeze also firing vs CB WR without TTM
- Do NOT use histogram direction; use price-break direction only

---

## Strategy 3: Keltner Channel ATR Expansion Breakout

**Source:** Linda Bradford Raschke, "Street Smarts" (1996), Chapter 7 "Holy Grail" setup;
  also Dr. Elder's "Come Into My Trading Room" (2002); tradingstrategyguides.com
**Applicability to VN30F1M:** HIGH

**Concept:**
Linda Raschke's "Holy Grail" is specifically: ADX(14) > 30 AND price pulls back to
touch/cross below the 20-period EMA (in uptrend) → buy on EMA touch. The Keltner Channel
provides the trend envelope. The setup identifies a pullback within a trending move,
exploiting mean-reversion to the EMA as entry point with the trend resuming.

**Parameters (Holy Grail version):**
- Trend filter: ADX(14) > 30 (strong trend)
- EMA: 20 period
- KC: EMA(20) ± ATR(14) * 1.5
- Signal (long): ADX > 30 AND 14-period ADX has risen (trend is accelerating) AND
  current bar low touches or crosses below EMA(20) from above
- Signal (short): mirror conditions
- Entry: limit order at EMA(20) or market on touch
- Stop loss: below prior swing low (longs) or above prior swing high (shorts)
- Target: upper KC band (ATR-based) = roughly 1.5x ATR from entry
- Alternate exit: trail at 1x ATR after target reached

**Alternative — Pure KC Breakout (Elder version):**
- KC: EMA(20) ± ATR(10) * 2
- Entry: close above upper band = long; close below lower band = short
- Confirmation: ADX(14) > 20 (any trend present)
- SL: re-cross below EMA(20) (for longs)
- Exit: opposite band touch OR session end
- Note: Elder's KC breakout is a TREND CONTINUATION tool, not a volatility expansion tool

**Reported backtest results:**
- Raschke Holy Grail: ~65% WR on daily futures; 60-63% on intraday 5m per community tests
- Requires strong ADX trend (>30) which filters ~60% of bars — reduces frequency
- After applying ADX>30 filter on VN30F1M: expect 0.5-1 signal/session (low frequency)
- Elder KC breakout: ~55% WR, higher frequency but lower R:R

**Intraday fit:**
- Sessions: yes
- Expected frequency: 0.5-1.5 signals/session (ADX filter is restrictive)
- Compatible with trailing exit: yes (Holy Grail has clear trail anchor at EMA)
- Works without direction prediction: NO — requires ADX trend direction; this is a
  trend-following entry, direction must be pre-determined by ADX slope

**Warning for VN30F1M:**
VN30F1M is often in choppy/non-trending regime, especially early session. ADX > 30 may
be too restrictive and fire only during momentum phases. Low frequency is a risk
(need min 1 signal/day).

**Integration idea:**
Use ADX(14) > 25 as a REGIME filter on CB signals rather than a standalone entry. When
ADX > 25 and CB fires in the direction of the existing trend (confirmed by EMA slope),
this is a trend-continuation CB — potentially higher WR than neutral CB. Test: CB WR
when ADX > 25 vs ADX < 25.

**Next step:**
Write `research/adx_regime_filter.py`:
- Compute ADX(14) on 5m data
- Split CB signals into: ADX>25 (trend) vs ADX<25 (chop)
- Report WR, PF separately for each regime
- Test threshold at 20, 25, 30 to find optimal cut
- Also test: ADX > threshold AND price above/below EMA(20) as trend direction filter

---

## Strategy 4: 2-Period RSI Connors Mean Reversion

**Source:** Larry Connors & Cesar Alvarez, "Short-Term Trading Strategies That Work" (2008);
  quantifiedstrategies.com/2-period-rsi-trading-strategy/
**Applicability to VN30F1M:** MEDIUM (mean-reversion, not volatility expansion)

**Concept:**
Use an extremely short RSI(2) as an overbought/oversold oscillator. RSI(2) < 10 after
price has closed below its 200-day MA (daily) or 50-bar MA (intraday) signals extreme
short-term oversold → buy. RSI(2) > 90 signals extreme overbought → sell short. This is
a MEAN REVERSION strategy (opposite of CB which is breakout/expansion). Connors found
RSI(2) has a very high WR (~65-70%) for equity index reversals.

**Parameters:**
- RSI period: 2 (very short, reacts within 1-2 bars)
- Oversold threshold: RSI(2) < 10 → long setup
- Overbought threshold: RSI(2) > 90 → short setup
- Trend filter (daily): price > SMA(200) for longs only (avoid shorting strong uptrends)
- Intraday equivalent: price > SMA(50, 5m) for longs, price < SMA(50, 5m) for shorts
- Entry: market order on bar close when RSI(2) threshold hit
- Exit: RSI(2) > 70 (for longs — exit when no longer oversold) OR RSI(2) < 30 (shorts)
- Alternative exit: 5-bar time exit (Connors default) — close position after 5 bars
- No hard stop loss in original system (relies on statistical edge); for VN30F1M must add
  SL = 1.5x ATR below entry

**Reported backtest results:**
- Connors SPY daily (1993-2008): 68% WR, avg gain 0.91% per trade
- Connors ES 5m intraday: ~61% WR, avg 0.3 pts per trade (before cost)
- After 0.96 pts cost on VN30F1M: marginal unless avg gain > 1.5 pts
- Frequency on 5m: RSI(2) < 10 fires on ~8-12% of bars → ~3-5 signals per session
  (potentially too many, would need additional filter)

**Intraday fit:**
- Sessions: yes
- Expected frequency: 3-5 potential signals/session (needs filtering to 1-2)
- Compatible with trailing exit: NO — this is a mean-reversion exit (take profit on bounce)
- Works without direction prediction: NO — this IS direction prediction (mean reversion)

**Warning for VN30F1M:**
This is a mean-reversion strategy. VN30F1M trending moves can persist for entire sessions,
making RSI(2) traps (signal fires but trend continues) a real risk. In trending markets,
RSI(2) < 10 can persist for 5+ bars before reverting. Must add ADX < 25 filter (no strong
trend) for safe application.

**Integration idea:**
RSI(2) is useful as a FILTER on CB signals, not as a standalone entry. CB fires best
when the market is "coiled" but not already in extreme territory. Test: CB WR when
RSI(14) is in range 30-70 vs RSI(14) outside this range. Also: CB fires after RSI(2)
briefly drops to oversold then recovers = compression + slight oversold recovery may
be a higher-confidence long CB setup.

**Next step:**
Write `research/rsi2_filter.py`:
- On CB signal bars, record RSI(2) and RSI(14) values
- Split signals: RSI(14) 30-70 (neutral) vs extremes
- Split signals: RSI(2) < 20 (oversold recovery) vs neutral vs overbought
- Report WR by RSI bucket
- Test RSI(2) standalone as a separate non-CB strategy with tight SL

---

## Strategy 5: MACD Histogram Divergence Scalp

**Source:** Alexander Elder, "Trading for a Living" (1993), "Come Into My Trading Room" (2002);
  Elder's Triple Screen system; community documentation at futures.io
**Applicability to VN30F1M:** MEDIUM

**Concept:**
Elder's "divergence" is NOT the classic price/oscillator divergence. It is a specific 3-bar
pattern in the MACD histogram: if price makes a lower low but the histogram makes a higher
low (bullish divergence), or price makes a higher high but histogram makes a lower high
(bearish divergence), a reversal is likely. This is a mean-reversion setup.

**Elder Triple Screen (full system):**
- Screen 1 (weekly/higher TF): trend direction via MACD-H or EMA slope
- Screen 2 (daily/current TF): oscillator divergence for timing
- Screen 3 (hourly/lower TF): precise entry on momentum

**Scalp version for 5m intraday:**
- MACD parameters: fast=12, slow=26, signal=9 (standard)
- Alternative for 5m: fast=5, slow=13, signal=5 (faster reaction)
- Bullish divergence: price low at bar -2 < price low at bar 0 AND histogram at bar -2
  < histogram at bar 0 (histogram is less negative = recovering)
- Entry: market order at next bar open after divergence confirmed
- Stop loss: below the lowest low of the divergence pattern (for longs)
- Target: MACD-H crosses zero (momentum turns positive) OR 1.5x ATR from entry
- Time exit: 10 bars max hold on 5m data

**Reported backtest results:**
- Elder's own research (equities, daily): ~60-65% WR on divergence signals
- 5m intraday futures community tests: ~52-56% WR (divergence is less reliable intraday)
- TradingView Pine strategy "MACD Divergence" (multiple authors): 50-58% WR on ES/NQ 5m
- After 0.96 pts cost: marginal on 5m; higher TF (15m) shows better results

**Intraday fit:**
- Sessions: yes
- Expected frequency: 1-3 divergence patterns/session on 5m
- Compatible with trailing exit: PARTIAL — exit on histogram zero-cross is the natural exit
- Works without direction prediction: NO — divergence is direction-specific

**MACD histogram as CB filter (the relevant application):**
The MOST applicable use for VN30F1M is not MACD divergence as a standalone entry, but
MACD histogram as a CB qualifier. If histogram is contracting toward zero during the CB
compression period → supports the "coiling" thesis. If histogram is flat and near zero
during compression → even better (true coil, energy building equally on both sides).

**Integration idea:**
At CB signal bar, check: is MACD histogram |value| < 0.3 (near zero/flat)?
If yes: market is in true balance, breakout could go either way with similar energy.
If no: MACD histogram suggests directional bias already present — may signal exhaustion
rather than compression. Test this as a filter on CB signals.

**Next step:**
Write `research/macd_cb_filter.py`:
- At each CB signal bar, compute MACD(12,26,9) histogram value
- Bucket: |hist| < 0.3 (balanced), 0.3-0.8 (mild bias), >0.8 (strong bias)
- Report CB WR by histogram bucket
- Also test: histogram contracting (|hist| shrinking over 3 bars) as a filter
- Expected: balanced histogram (near zero) should have higher WR for CB

---

## Strategy 6: Stochastic + Volume Momentum Combo

**Source:** Jake Bernstein, "Short-Term Trader's Manual" (1999); tradingstrategyguides.com
  "Stochastic Indicator Strategy"; also Barry Burns "TradeFinder" methodology
**Applicability to VN30F1M:** MEDIUM

**Concept:**
Combine fast Stochastic oscillator (5,3,3) for short-term momentum timing with volume
confirmation. Entry when Stochastic oversold (%K < 20) AND volume on the signal bar is
above average (> 1.2x 20-bar average volume) AND price is above VWAP (for longs). This
uses volume surge as confirmation that the oversold condition is hitting real buying interest
rather than a slow drift into oversold.

**Parameters:**
- Stochastic: K=5, D=3, smooth=3 (fast version)
- Oversold: %K < 20 AND %K crosses above %D while below 20 (stoch cross)
- Overbought: %K > 80 AND %K crosses below %D while above 80
- Volume filter: bar volume > 1.2x SMA(volume, 20)
- VWAP filter: price > VWAP for longs, price < VWAP for shorts (optional)
- Entry: next bar open after stoch cross + volume confirmation
- Stop loss: 1.0x ATR(14) below entry (longs)
- Target: Stochastic %K reaches 80 (mean reversion complete) OR 2x ATR from entry
- Trail: 0.8x ATR after 1.5x ATR MFE (tighter trail for this scalp strategy)

**Reported backtest results:**
- Bernstein's original (daily futures): 58-62% WR, avg 0.7 pts/trade
- Community 5m tests (ES, NQ): 50-56% WR before costs; 45-52% after typical costs
- Volume filter improvement: +3-5% WR vs stochastic alone per multiple community tests
- After 0.96 pts cost on VN30F1M: borderline; needs avg gain > 2.0 pts to be viable

**Intraday fit:**
- Sessions: yes
- Expected frequency: 2-4 signals/session (oversold/overbought touches common on 5m)
- Compatible with trailing exit: YES (tight trail works well for scalp)
- Works without direction prediction: NO — stochastic cross is direction-specific

**VWAP component — the most valuable piece:**
VWAP is the institutional benchmark. VN30F1M during trending sessions will stay above
VWAP for the entire AM session. Testing: CB signals that fire with price > VWAP (longs)
or price < VWAP (shorts) may have higher WR than CB signals against VWAP.

**Integration idea:**
VWAP alignment as CB filter: compute VWAP from session open. If CB long signal fires
AND price > VWAP: high-confidence. If CB long fires AND price < VWAP: reduce confidence
or skip. This is a direction-free filter in the sense that it just checks alignment with
dominant institutional flow.

**Next step:**
Write `research/vwap_cb_filter.py`:
- Compute session VWAP from 9:00 for AM session, from 13:00 for PM session
- At each CB signal: check price vs VWAP alignment (long above VWAP, short below VWAP)
- Report WR: aligned vs counter-VWAP CB signals
- Also report frequency: how often does CB fire counter-VWAP? Skip those?
- Expected: aligned CB signals should have meaningfully higher WR

---

## Strategy 7: Central Pivot Range (CPR) Bounce + Breakout

**Source:** Tom DeMark pivot system; B.C.P. derivatives trading community; Indian futures
  trading community (CPR is extensively documented for NSE/BSE derivatives);
  Investopedia "Pivot Point Trading Strategy"
**Applicability to VN30F1M:** HIGH

**Concept:**
CPR (Central Pivot Range) is a 3-line pivot zone: Bottom Central Pivot (BC), Pivot Point
(PP), and Top Central Pivot (TC), all derived from prior day H, L, C. A narrow CPR (TC
close to BC) on day N+1 predicts a trending day — price will break out of the CPR and
trend. A wide CPR predicts mean-reversion day — price will bounce around CPR. This is a
VOLATILITY PREDICTION tool similar in spirit to CB.

**Parameters:**
- Calculated from PRIOR DAY: H_prev, L_prev, C_prev
- PP = (H_prev + L_prev + C_prev) / 3
- BC = (H_prev + L_prev) / 2
- TC = (PP - BC) + PP
- CPR Width = TC - BC
- CPR Width relative to ATR: narrow = CPR_width < 0.3x ATR → trending day expected
  wide = CPR_width > 0.8x ATR → sideways/reversal day expected
- Standard pivot levels: R1 = 2*PP - L_prev, S1 = 2*PP - H_prev
  R2 = PP + (H_prev - L_prev), S2 = PP - (H_prev - L_prev)

**Entry rules — CPR Breakout (trending day):**
- Condition: CPR Width < 0.3x prior day ATR
- Long entry: price breaks above TC + 0.1
- Short entry: price breaks below BC - 0.1
- Stop loss: opposite side of CPR (below BC for longs, above TC for shorts)
- Target 1: R1 for longs, S1 for shorts
- Target 2: R2 for longs, S2 for shorts
- Trail after T1: 1x ATR trail

**Entry rules — CPR Bounce (wide CPR / sideways day):**
- Condition: CPR Width > 0.8x prior day ATR
- Long: price approaches S1 from above, bounces up with a reversal bar
- Short: price approaches R1 from below, reverses with a reversal bar
- Target: PP (the middle)
- SL: 0.5 pts below S1 (longs) or above R1 (shorts)

**Reported backtest results:**
- Indian NSE Nifty futures (similar to VN30F): narrow CPR breakout ~63-67% WR, PF ~2.0
- Wide CPR bounce: ~58-62% WR (mean reversion is slightly less reliable)
- Combined CPR system on 5m: well-documented WR ~62% in Indian derivatives community
- CPR width as a DAY CLASSIFICATION tool: accuracy of narrow CPR predicting trending day
  is reported at ~65% in multiple Indian trading strategy books

**Intraday fit:**
- Sessions: YES — CPR calculated overnight before 9:00 open, used all day
- Expected frequency: 1-3 CPR breakout signals in AM, 1-2 in PM
- Compatible with trailing exit: YES — trail after hitting R1/S1 targets
- Works without direction prediction: PARTIAL — CPR breakout requires knowing WHICH
  side of CPR to trade, but this is determined by price action (whichever side breaks first)

**This is a DAY-LEVEL volatility predictor — complementary to CB (bar-level):**
CPR width predicts whether the WHOLE DAY will trend or chop. CB detects real-time
compression within the day. Combining both: on narrow CPR days (trending day predicted),
CB signals have higher probability of sustained breakout. On wide CPR days (choppy predicted),
CB signals may be fade-worthy (mean reversion more likely).

**Integration idea (DAY-LEVEL REGIME):**
Classify each trading day as "CPR Narrow" (CPR_width < 0.3x ATR) or "CPR Wide" (>0.8x ATR).
Test CB WR separately on each day type. If narrow CPR days show WR > 70% for CB, this
is a powerful pre-market filter. Can reject trades on wide CPR days or flip to fade mode.

**Next step:**
Write `research/cpr_day_filter.py`:
- Load daily H/L/C for VN30F (from OHLCV) to compute prior-day CPR
- Compute CPR width as fraction of prior-day ATR
- Classify trading days: narrow / neutral / wide CPR
- Join with CB signal log: report WR by CPR day type
- Also plot: CPR levels on each day to verify they act as S/R for 5m price
- Expected: CB WR on narrow CPR days > average CB WR

---

## Strategy 8: Opening Range Breakout (ORB) — Gap + First Bar

**Source:** Toby Crabel, "Day Trading with Short-Term Price Patterns" (1990) — the original
  ORB research; Mark Fisher "ACD Method"; Jeff Cooper "Hit & Run Trading";
  quantifiedstrategies.com/opening-range-breakout/
**Applicability to VN30F1M:** HIGH

**Concept:**
The Opening Range (OR) is the high-low range of the first N bars of the session (typically
first 15 or 30 minutes). A breakout above OR high = long entry; below OR low = short entry.
The compression within the opening range represents market participants discovering fair value
after overnight gap. Once the range is established and broken, the new direction is confirmed.
This is structurally identical to CB but using the first session bars as the "compression zone."

**Parameters (Crabel original):**
- OR period: first 1 bar on daily; for 5m intraday → first 3 bars (15 min) or 6 bars (30 min)
- Long entry: price crosses above OR_high + small buffer (0.1 pts)
- Short entry: price crosses below OR_low - small buffer
- Stop loss: opposite extreme of OR (OR_low - buffer for longs)
- Target: OR_high + (OR range * 1.0) for first target (1:1 expansion)
  OR_high + (OR range * 2.0) for second target
- Time exit: close position at AM session end (11:25)
- Also tested: if ORB fires within 30 min of session close, skip the signal

**ORB with Gap filter (Fisher ACD style):**
- Prior day close vs today's open: gap > 0.3% = "gap day"
- On gap up days: only take ORB LONG signals (gap direction has follow-through 65% of time)
- On gap down days: only take ORB SHORT signals
- On no-gap days: take both directions (whichever triggers first)
- Additional: if price fills the gap before triggering ORB, the signal is weakened — skip

**Reported backtest results:**
- Crabel original (daily, stocks): 60-65% WR on ORB breakout
- 5m intraday ORB (15-min OR): ~58-63% WR on ES/NQ futures per multiple studies
- quantifiedstrategies.com ORB study (SPY ETF): 55% WR without gaps, 62% WR with gap
  alignment filter
- Frequency: 1-2 signals per session (only one OR per session start)
- VN30F1M specific: AM session starts at 9:00; OR = bars 9:00-9:14 (3 bars on 5m)

**Gap fill specifically:**
- VN30F1M gaps (prior close vs 9:00 open) of > 1.0 pt are common
- Gap fill rate (gap closed intraday) on VN30F: approximately 60-65% for gaps < 1.5 pts
- Strategy: enter at open in direction of gap fill, exit when gap filled
- SL: extension beyond OR high/low (gap makes new extreme)
- This is counter-trend to gap; requires tight discipline

**Intraday fit:**
- Sessions: YES — perfect fit. Signal fires in first 15-30 min of session
- Expected frequency: exactly 1 per session (2/day AM+PM) but PM OR = 13:00-13:14
- Compatible with trailing exit: YES — trail after first target hit
- Works without direction prediction: YES — wait for whichever side breaks the OR first

**This is essentially a STRUCTURED CB at session open:**
ORB and CB share the same volatility expansion logic. ORB is a FIXED TIME version
(first N bars) while CB is an ANY TIME version (any 3 consecutive narrow bars).
ORB fires once per session and tends to be high quality because session opens are
naturally compressed (price discovery phase).

**Integration idea:**
Add ORB logic to existing scanner:
1. At 9:15 (after 3 bars of 5m data), compute OR = [9:00 high, 9:00-9:04, 9:05-9:09, 9:10-9:14]
2. Set long_trigger = max(OR highs) + 0.1, short_trigger = min(OR lows) - 0.1
3. Arm both triggers for the AM session
4. If CB also fires AND the breakout direction aligns with ORB direction → dual confirmation
   signal
5. Compare: ORB standalone WR vs ORB+CB co-occurrence WR

**Next step:**
Write `research/orb_strategy.py`:
- Identify session opens (9:00 AM, 13:00 PM)
- Compute OR from first 3 bars of each session (15 min)
- OR range: OR_high = max(high[0:3]), OR_low = min(low[0:3])
- Simulate breakout entry with SL at opposite extreme
- Report WR, PF, avg MFE/MAE for pure ORB
- Also test: OR width as a filter (narrow OR on narrow CPR day = very high confidence)

---

## Summary Table — Ranking by VN30F1M Applicability

| # | Strategy | Applicability | Volatility-based? | Frequency | Key Use |
|---|----------|--------------|-------------------|-----------|---------|
| 1 | NR7 Breakout | HIGH | YES | 2-4/session | Standalone or CB+NR7 combo |
| 2 | TTM Squeeze | HIGH (vol only) | YES | 1-2/session | CB confirmation filter |
| 3 | CPR Day Filter | HIGH | YES (day-level) | 1/day classifier | Pre-market regime filter |
| 4 | ORB (Opening Range) | HIGH | YES | 1/session | Session-open CB equivalent |
| 5 | KC/Holy Grail | HIGH (as filter) | PARTIAL | 0.5-1.5/session | ADX regime filter for CB |
| 6 | VWAP Alignment | MEDIUM-HIGH | NO (direction) | Continuous | CB direction confidence |
| 7 | MACD Histogram | MEDIUM | PARTIAL | 1-3/session | CB balance filter |
| 8 | Stochastic+Vol | MEDIUM | NO | 2-4/session | VWAP component only |
| 9 | RSI(2) | MEDIUM | NO | 3-5/session | CB extreme filter |
| 10 | MACD Divergence | LOW-MEDIUM | NO | 1-3/session | Direction anti-pattern risk |

---

## Priority Backtest Queue

In order of expected value and ease of implementation:

**Priority 1 — Test immediately (high confidence in edge):**
1. `research/cpr_day_filter.py` — CPR width as day-type classifier for CB
2. `research/orb_strategy.py` — ORB as first-of-session CB signal
3. `research/nr7_breakout.py` — NR7 as CB extension/confirmation

**Priority 2 — Test as CB filters:**
4. `research/vwap_cb_filter.py` — VWAP alignment on CB signals
5. `research/adx_regime_filter.py` — ADX threshold for trend-day CB
6. `research/ttm_squeeze_filter.py` — TTM co-occurrence with CB

**Priority 3 — Test standalone (lower confidence):**
7. `research/macd_cb_filter.py` — MACD histogram balance at CB bar
8. `research/rsi2_filter.py` — RSI(2) extreme filter on CB signals

---

## Composite Combo Idea: "Triple Confirmation" Signal

From the research above, the HIGHEST confidence intraday breakout signal for VN30F1M
would be a composite requiring:

1. **Day-level:** Narrow CPR (CPR width < 0.3x ATR) → trending day expected
2. **Bar-level:** CB compression fires (existing strategy) OR NR7 bar identified
3. **Momentum-level:** MACD histogram near zero (< 0.3 abs value) at compression bar
4. **Flow-level:** Price on correct side of VWAP (long above VWAP, short below)
5. **Trigger:** ORB OR direction (whichever side of OR/compression breaks first)

Each condition alone: ~55-65% WR. Combined (if filters are independent): could push
toward 70%+ WR. The risk is signal frequency drops to < 1/day. Needs careful backtesting
to confirm independence of filters (they may be correlated and not truly multiplicative).

**Implementation note:** Test first with 2-condition combos before requiring all 5. Start
with CPR + CB (two independent sources: daily and intraday), then add NR7, then VWAP.

---

## Notes on Data Requirements

- CPR requires prior-day daily OHLC → available via vnstock daily endpoint
- VWAP requires tick volume per 5m bar → available in standard OHLCV
- NR7 requires only OHLCV → computed from existing 5m data (144d available)
- TTM Squeeze (BB + KC) requires only OHLCV → available
- ORB requires knowing session open time → use 9:00 and 13:00 timestamps
- ATR-based pivots need intraday OHLCV → available
- None of these require tick data or Level 2 — all workable with existing data pipeline

---

*File generated: 2026-06-05. Update after each backtest cycle.*
