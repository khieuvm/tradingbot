# VN30F1M Intraday Futures Trading Bot

## Project Overview

Automated trading bot for VN30F1M (Vietnam VN30 Index Futures, front-month contract). Intraday only — no overnight holds. Sessions: 9:00-11:30 and 13:00-14:30 (Vietnam time, UTC+7).

## Quick Start (Deploy to New Machine)

### 1. Install dependencies

```bash
pip install vnstock dnse pandas pandas_ta pyyaml python-dotenv
```

### 2. Create `.env` file

```env
TELEGRAM_BOT_TOKEN=<bot token from @BotFather>
TELEGRAM_CHAT_ID=<your chat ID>

DNSE_API_KEY=<from DNSE developer portal>
DNSE_API_SECRET=<from DNSE developer portal>
DNSE_ACCOUNT_NO=<derivative sub-account ID, NOT custody code>

# Gmail IMAP for auto OTP
DNSE_OTP_EMAIL=<gmail address registered with DNSE>
DNSE_OTP_APP_PASSWORD=<Gmail App Password from https://myaccount.google.com/apppasswords>
```

**Important:**
- `DNSE_ACCOUNT_NO` is the **sub-account ID** (e.g. `0001617524`), NOT the custody code (e.g. `064C698687`). Get it via `client.accounts.list()` → look for `derivative_account=True`.
- Gmail App Password requires 2FA enabled on Google account.
- DNSE trading token is valid for **8 hours** — bot auto-authenticates on startup.

### 3. Run

```bash
# Signal-only mode (no real orders)
python scanner.py --no-trade

# Live trading (auto-auth + real DNSE orders)
python scanner.py

# Single scan (test)
python scanner.py --once --no-trade
```

### 4. Verify

- Bot prints `[DNSE] Ready. Contract: 41I1G6000` on startup
- Telegram receives "CB Scanner Started" message
- During trading hours: scans every 10s (position update) / 60s (CB signal scan)
- After 14:29: auto-closes all positions + sends EOD report

## Architecture

```
scanner.py               — Main loop: 10s position update + 60s CB scan
combos/base.py           — Abstract base class for all combo strategies
combos/cb.py             — CB Compression Breakout detection (OOP)
combos/__init__.py       — Combo registry (get_combo by name)
src/portfolio_manager.py — Session-aware position mgmt, adaptive exit, ATR trailing
src/position_manager.py  — Legacy single-position manager (kept for compatibility)
src/dnse_auth.py         — Auto OTP via Gmail IMAP → DNSE trading token
src/dnse_executor.py     — Real order placement via DNSE API (symbol: 41I1G6000)
src/data_fetcher.py      — OHLCV data from vnstock/KBS
src/notifier.py          — Telegram notifications (entry/exit/EOD/errors)
src/trade_logger.py      — Daily JSONL trade logs + summary generation
backtest/engine.py       — CB backtest engine (trail sweep, simulation)
research/                — Analysis tools (placeholder)
config.py                — Environment variable loader
strategy_config.yaml     — Session params, adaptive exit, combo_tf_map (CB: [5m])
```

### Data Flow

```
vnstock/KBS API → data_fetcher.py → scanner.py → combo.detect(df_5m)
                                        ↓
                              portfolio_manager.py (manage positions)
                                        ↓
                              dnse_executor.py → DNSE API (place orders)
                                        ↓
                              notifier.py → Telegram (alerts)
                              trade_logger.py → logs/ (JSONL)
```

### Key Design Decisions

- **Single strategy: CB (Compression Breakout)** — 3-bar squeeze < 0.7×ATR → breakout
- **Data source:** vnstock/KBS for OHLCV (DNSE has no OHLCV endpoint)
- **Order execution:** DNSE direct API (`dnse` package), symbol `41I1G6000`
- **Auth:** Auto OTP via Gmail IMAP — fully unattended after .env setup
- **Dual loop:** 10s for fast SL/TP checking, 60s for CB signal detection
- **1 contract max:** 40M account, margin ~37M, only 3M buffer

## CB Strategy Details

### Signal Detection (`scanner.py:detect_cb_compression`)
- **Compression:** max range of 3 bars before signal bar < 0.7 × ATR(14) on 5m
- **ATR filter:** 2.5 ≤ ATR ≤ 4.5 (rejects low-vol and high-vol)
- **RSI filter:** RSI(14) < 70 (rejects overbought)
- **Time filter:** AM 09:15–10:45, PM 13:15–14:15
- **Dedup:** 25 min between signals (= 5 bars, matches backtest)
- **Regime:** skip CB in VOLATILE regime (ATR ratio > 1.5×SMA50)
- **Direction:** BUY trigger = signal bar high + 0.1, SELL trigger = low - 0.1
- **Next-bar confirmation:** 2-scan expiry if trigger not hit

### Exit Logic (`portfolio_manager.py:update_prices`)

| Exit type | Trigger | Notes |
|-----------|---------|-------|
| SESSION | AM ≥ 11:25, PM ≥ 14:25 | Matches backtest cutoffs |
| MAX_HOLD | elapsed ≥ bars×5 min (AM 120min, PM 60min) | Time-based, not tick-based |
| ADAPT_EXIT | AM + pre_move/ATR > 0.8 + MFE ≥ 4pts + PnL ≥ 3.2pts | Close immediately |
| Adapt SL lock | AM + above but PnL < 3.2pts | Move SL to entry+2pts |
| BE | MFE ≥ 4pts when entry ATR ≥ 3.5 | SL → entry |
| TRAIL | MFE ≥ 5pts (AM) / 4pts (PM) | SL = best_price − trail_mult×ATR |
| Trail tighten | MFE ≥ 8pts when ATR ≥ 3.5 | trail_mult → 1.2× (live-only) |
| PM tighten | PM, ≤14 min to 14:29 | SL → close − 0.5×ATR |
| SL | price hits stop | TRAIL or SL label depending on tp1_hit |
| TP | price hits 4×ATR target | Live-only (backtest has no fixed TP) |
| EOD | 14:29 | Force-close all via scanner.py |

## Key Parameters

| Parameter | Value |
|-----------|-------|
| Strategy | CB (Compression Breakout) / 5m only |
| Backtest result | WR 64-65%, PF 4.5, +5.99-6.05/d (129d); +6.34/d with adaptive exit |
| Scan interval | 10s (position) / 60s (signals) |
| Cost per trade | 0.96 pts (slippage 0.5 + commission 0.46) |
| DNSE symbol | 41I1G6000 (VN30F1M KRX format) |
| Max contracts | 1 (40M account) |
| AM SL | 1.2×ATR, trail@5pts/2.0×ATR, max 120 min, exit 11:25 |
| PM SL | 1.0×ATR, trail@4pts/1.5×ATR, max 60 min, exit 14:25 |
| Adaptive exit | AM + pre_move/ATR > 0.8 + MFE≥4pts → close if PnL≥3.2, else SL→entry+2 |
| BE trigger | MFE ≥ 4pts when entry ATR ≥ 3.5 → SL to entry |
| Trail tighten | MFE ≥ 8pts when ATR ≥ 3.5 → trail 1.2×ATR |
| EOD force close | 14:29 (must close before 14:30) |
| Trading token | Valid 8h, auto-refresh at startup |

## DNSE API Details

| Tier | Rate/Hour | Used For |
|------|-----------|----------|
| High | 50,000 | Place/cancel orders |
| Standard | 10,000 | Account, positions |
| Low | 1,000 | OHLC (not used — we use vnstock) |
| Minimal | 100 | OTP, trading token |

Rate limit exceeded → HTTP 429. Bot retries next cycle.

## Conventions

- All analysis output in Vietnamese or English per user preference
- Cost: **0.96 pts/trade** — NEVER use 1.74
- Regime filter: skip CB when VOLATILE (5m ATR_ratio > 1.5×SMA50)
- Minimum sample: 30 trades for any statistical claim
- Walk-forward: 60d IS / 30d OOS, require OOS PF > 1.3
- Monte Carlo: 100 permutations, require p < 0.05
- Direction from breakout (H+0.1 / L-0.1), confirmed on next bar
- AM/PM always tested separately — they have different dynamics
- Backtest reference: `bt_trail_sweep.py` — never modify without re-validating

## Critical Rules

1. **No overnight holds** — all positions must close by 14:28
2. **Cost-aware** — every strategy must be profitable AFTER 0.96 pts cost/trade
3. **Regime-aware** — skip CB in VOLATILE regime
4. **Robust validation required** — no strategy goes live without walk-forward OOS confirmation
5. **Signal decay tracking** — auto-disable CB if rolling WR < 40% over 10 trades
6. **Adaptive exit** — AM session + pre_move/ATR > 0.8 + MFE≥4pts (proven +0.29/d improvement)
7. **Data limits** — API: 5m ~144d, 3m/1m ~28d. Holiday fix: filter from 2026-05-02
8. **Single contract** — max 1 contract, 40M account with only 3M buffer above margin
9. **Session exits are hard** — AM 11:25, PM 14:25 — no position survives its session

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `FORBIDDEN: account not found` | Wrong account number. Use sub-account ID from `client.accounts.list()`, not custody code |
| OTP not found in email | Check Gmail App Password, check DNSE sender (`noreply@mail.dnse.com.vn`) |
| `HTTP 429` | Rate limited. vnstock fetches too fast — increase SCAN_INTERVAL |
| No CB signals during market hours | Check ATR (must be 2.5-4.5) and time window (AM 09:15-10:45, PM 13:15-14:15) |
| Position not closing at session end | Check SESSION exit in `portfolio_manager.update_prices()` |
| `vnstock` symbol error | VN30F1M auto-converts to 41I1G6000 (KRX format) internally |
| Positions surviving past 11:30 | SESSION exit at 11:25 should catch this; session-change exit at 13:00 as safety net |

## ML Module (Meta-Labeling Signal Filter)

### Status: INFRASTRUCTURE COMPLETE — Chờ Validation

ML module đã build xong, **chưa bật live** vì OOS validation chưa đủ mạnh.

### Kiến trúc

```
ml/
  features.py           — 25 features tính từ 5m OHLCV tại signal bar
  labeler.py            — Chạy backtest → extract features + label (win/loss)
  train_meta_label.py   — Train LightGBM + purged walk-forward CV
  model.py              — MLFilter class cho scanner integration
  config.py             — Default ML params
  data/cb_trades_labeled.csv   — 222 trades labeled (180d backtest)
  models/meta_label_latest.pkl — Trained model
  reports/validation_report.json
```

### Cách chạy

```bash
# 1. Generate dataset (chạy lại khi có thêm data)
python -m ml.labeler

# 2. Train + validate
python -m ml.train_meta_label

# 3. Shadow mode (log predictions, KHÔNG veto)
python scanner.py --ml-shadow --no-trade
```

### Kết quả validation hiện tại (2026-06-08)

- Dataset: 222 trades, WR 75.2%, 128 ngày
- OOS Fold: 1 fold duy nhất (165 train / 41 test)
- **ML chưa thêm giá trị** — tất cả thresholds cho Net PnL < baseline
- Lý do: CB WR quá cao (75%), chỉ 55 losers → ML khó phân biệt
- Top features: atr_14, ret_13, vol_ratio, range_pct, adx, compression_depth

### TODO — Công Việc Tiếp Theo

1. **[ĐANG CHỜ] Shadow mode 10-15 ngày** — chạy `scanner.py --ml-shadow` song song live, thu thập predictions
2. **[ĐANG CHỜ] Thêm data** — cần 300+ trades (thêm 2-3 tháng) để có 2+ OOS folds đáng tin
3. **[CẦN LÀM] Thử LOOCV** — Nếu muốn validate nhanh hơn, sửa `train_meta_label.py` để chạy purged LOOCV trên toàn bộ 222 trades (code đã có, chỉ cần force call)
4. **[CẦN LÀM] Thêm features từ data mới** — orderbook imbalance, foreign flow, basis (nếu có API)
5. **[PHASE 2] Exit optimization** — ML predict optimal trail/BE per-trade (sau khi Phase 1 validated)
6. **[PHASE 3] Standalone ML signals trên 1m** — Chờ 120+ ngày 1m data (~32,400 bars)

### Quyết Định Đã Chốt

- ML **augment** CB, KHÔNG replace
- Shadow mode **bắt buộc** trước khi go live
- Kill switch: auto-disable nếu ML-filtered PnL < unfiltered 10 ngày liên tục
- Retrain monthly khi có đủ data mới

### Dependencies Bổ Sung (cho ML)

```bash
pip install lightgbm scikit-learn
```

Nếu không install LightGBM, code tự fallback sang ExtraTreesClassifier (sklearn built-in).
