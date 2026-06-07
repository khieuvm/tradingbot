# Combo Catalog — 20 Intraday Strategies for VN30F1M

Researched 2026-06-05 from GitHub, Reddit, academic papers, trading education sites.
All strategies are intraday-only, work on 5m candles, no overnight holds.

## Summary Table

| # | Name | Type | WR (reported) | PF | Freq/day | Direction-free |
|---|------|------|---------------|-----|----------|---------------|
| 1 | NR7 Breakout | Entry | 55-68% | 1.6-2.4 | 2-4 | Yes |
| 2 | NR4 Breakout | Entry | 56-62% | 1.5-2.0 | 3-5 | Yes |
| 3 | ORB (Opening Range Breakout) | Entry | 58-65% | 1.8-2.4 | 1-2 | Yes |
| 4 | Inside Bar Double IB | Entry | 62-65% | 2.4 | 1-2 | Yes |
| 5 | TTM Squeeze Fire | Entry | 53-58% | 1.9-2.2 | 1-3 | Yes |
| 6 | Multi-TF Compression (15m+5m) | CB Filter | 68-72% | 2.0-2.8 | 0.5-1 | Yes |
| 7 | Donchian Channel Breakout | Entry | 55-60% | 1.5-2.0 | 0.5-1 | Yes |
| 8 | VWAP SD Band Breakout | Entry | 58-64% | 1.4-1.6 | 0.5-1.5 | Yes |
| 9 | VWAP Mean Reversion | Entry | 58-67% | 1.5-1.9 | 0.5-1.5 | Yes |
| 10 | VWAP Trend Pullback | Entry | 54-57% | 1.7-2.1 | 1-2 | Partial |
| 11 | Gap Fade | Entry | 61-68% | 1.5-1.8 | 0.5-1.5 | No (gap dir) |
| 12 | CPR Day Filter + Breakout | Entry/Filter | 63-67% | 2.0 | 1-3 | Yes |
| 13 | ADX Compression Filter | CB Filter | +2-3% WR | - | filter | Yes |
| 14 | Choppiness Index Filter | CB Filter | +2-5% WR | - | filter | Yes |
| 15 | RSI(2) Mean Reversion | Entry | 61-68% | 1.4-1.6 | 3-5 | No |
| 16 | ADX Pullback to EMA | Entry | 54-58% | 1.6-1.9 | 0.5-1.5 | Partial |
| 17 | Intraday Momentum (First 30m) | Filter | 55-58% | 1.6-1.8 | 1-2 | No |
| 18 | Volume Profile POC Bounce | Entry | 58-67% | 2.1 | 1-3 | Yes |
| 19 | ATR Ratio Regime Tiers | CB Filter | +15-25% Sharpe | - | filter | Yes |
| 20 | MACD Histogram Balance | CB Filter | +2-3% WR | - | filter | Yes |

## Priority Tiers

### Tier 1 — Highest confidence, easiest to implement
1. **NR7 Breakout** — Same compression family as CB, rank-based
2. **NR4 Breakout** — Higher frequency variant
3. **ORB** — Session-open compression, complementary to CB
4. **Inside Bar Double IB** — Tightest compression, highest reported WR
5. **Multi-TF Compression** — Direct CB enhancement, no new data needed
6. **CPR Day Filter** — Day-level volatility predictor

### Tier 2 — Medium priority, more complex
7. **TTM Squeeze Fire** — Dual compression (BB inside KC)
8. **Gap Fade** — Standalone pre-CB window signal
9. **VWAP Mean Reversion** — Ranging-day strategy
10. **Donchian Channel** — Trend continuation
11. **ADX Compression Filter** — CB WR booster
12. **Choppiness Index Filter** — CB WR booster

### Tier 3 — Lower priority or direction-dependent
13-20. Various filters and direction-dependent strategies
