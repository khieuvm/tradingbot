# VN30F1M Trading Scanner

Real-time signal scanner for VN30F1M futures with multi-timeframe confirmation, dynamic entry optimization, and portfolio management.

## Performance (120-day backtest)

| Metric | Value |
|--------|-------|
| Win Rate | 83.1% |
| Total P&L | +172.7 pts (+17.3M VND) |
| Avg P&L/trade | +2.66 pts |
| Profit Factor | 2.85 (D, J) |
| Total Trades | 65 |

### Active Combos

| Combo | TF | WR | P&L | PF |
|-------|----|----|-----|-----|
| D: AM Mid Trend | 5m | 86.5% | +84.5 | 2.85 |
| J: RSI Momentum | 5m | 82.4% | +55.9 | 2.85 |
| X: Big Move Catcher | 3m, 5m | 50-75% | +23.9 | 2.28-3.84 |
| V: VWAP Dip Bounce | 5m | 100% | +8.4 | 837 |

## Key Features

### Entry System
- **Next-bar confirmation**: Signal fires → wait for next bar to break signal bar's high/low. Filters ~50% of traps (reduces SL rate from 75% to 46%).
- **Dynamic limit offset**: Entry pullback based on signal bar strength:
  - Strong candle (body>70%) or high volume (>3x avg): 0.3 ATR offset
  - Normal: 0.5 ATR offset
  - Weak candle (body<50%): 0.8 ATR offset
- **Min TP filter**: Skip trades where TP distance < 3 points

### Risk Management
- Per-combo SL/TP/trailing parameters (strategy_config.yaml)
- Trailing stop with configurable offset & step per combo
- Exhaustion filter: skip if 2+ consecutive same-direction signals
- Max hold timeout per combo

### Signal System (src/signals.py)
- 77 conditions across trend, momentum, reversal, scalp categories
- Gate system: hard filters (EMA200 bias, EMA slope, strong candle, prev bar)
- Combo presets with primary/confirm/gate architecture

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Live scanner (continuous)
python scanner.py

# Single scan
python scanner.py --once

# Backtest (120 days)
python backtest_120d.py
```

## Configuration

- `strategy_config.yaml`: Combo risk params, TF mapping, trailing config
- `config.py`: API keys, Telegram bot config

## Architecture

```
scanner.py          - Main scanner loop with confirmation logic
src/signals.py      - Signal generation engine (77 conditions)
src/data_fetcher.py - vnstock API wrapper
src/portfolio_manager.py - Multi-position portfolio management
src/position_manager.py  - Single position management
src/notifier.py     - Telegram alerts
src/trade_logger.py - Trade logging
backtest_120d.py    - Focused backtest with optimized params
```
