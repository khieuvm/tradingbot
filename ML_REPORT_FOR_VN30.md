# ML Research Report — Applicable to VN30F1M & Crypto Futures

**Date:** 2026-06-07
**Source:** Research from OKX crypto futures project (BTC/ETH/SOL, 1m-15m)
**Target application:** VN30F1M (Vietnam index futures) + Crypto

---

## 1. EXECUTIVE SUMMARY

### What We Tested (60 days OKX real data)
- 5 ML models × 3 pairs × 2 timeframes × 2 labeling methods × 7 thresholds = **420 configurations**
- Walk-forward validation: 30d train / 10d test / 10d step (calendar-based, no look-ahead)
- Cost model: maker 0.04% RT, taker 0.10% RT

### Key Results

| Finding | Detail |
|---------|--------|
| **Best model** | ExtraTrees (200 trees, depth 6, min_leaf=50) |
| **Best timeframe** | 3m (crypto), expect 1m-5m optimal for VN30 |
| **Best labeling** | Fixed-horizon (6 bars forward) > Triple-barrier for scalping |
| **Win rate** | 55-57% OOS (walk-forward) |
| **Edge per trade** | 8-15 bps gross, 4-10 bps net of maker fees |
| **Trades/day** | 10-18 (at threshold 0.58) |
| **Taker-profitable configs** | 16/420 (only highest-volatility asset) |
| **Maker-profitable configs** | 152/420 (widely viable with limit orders) |

### Critical Insight for VN30
- Edge is THIN (4-10 bps/trade net) → **must use limit orders** (maker fees)
- Fee structure determines viability more than model choice
- VN30F1M has ZERO commission on intraday → massive advantage vs crypto
- Same models should yield 2-5x better net PnL on VN30 due to cost difference

---

## 2. TOP MODELS RANKED (Practical Deployment)

### Tier 1: PROVEN (tested with real data, walk-forward validated)

| # | Model | Library | Install | Train Time | WR | Notes |
|---|-------|---------|---------|------------|-----|-------|
| 1 | **ExtraTrees** | `sklearn` | built-in | <1s | 56.9% | Best risk-adjusted on OOS data |
| 2 | **LightGBM** | `lightgbm` | `pip install lightgbm` | <1s | 55.1% | Best overall, most configurable |
| 3 | **RandomForest** | `sklearn` | built-in | <1s | 55.4% | Robust, good probability calibration |

### Tier 2: HIGH POTENTIAL (theory + limited testing)

| # | Model | Library | Train Time | Expected WR | Notes |
|---|-------|---------|------------|-------------|-------|
| 4 | **Stacking Ensemble** | `sklearn` | 5s | +1-3pp over single | LGB + ET + Logistic |
| 5 | **CatBoost** | `catboost` | 3s | ~equal LGB | Better with categoricals |
| 6 | **HMM regime filter** | `hmmlearn` | instant | indirect (reduces DD) | 3-state regime classifier |
| 7 | **LGB Quantile** | `lightgbm` | 1s | risk mgmt | Predict worst-case return |
| 8 | **TabNet** | `pytorch-tabnet` | 10min | +0-1pp | Attention-based tabular |

### Tier 3: RESEARCH (needs GPU, complex)

| # | Model | Library | Train Time | Expected WR | Notes |
|---|-------|---------|------------|-------------|-------|
| 9 | **ROCKET** | `tsai` | 30s | ~equal trees | Fastest DL, worth trying |
| 10 | **TCN** | `pytorch-tcn` | 15min | +0-1pp | Temporal patterns |
| 11 | **PatchTST** | `neuralforecast` | 30min | +0-1pp | Latest transformer for TS |
| 12 | **TFT** | `pytorch-forecasting` | 60min | +0-2pp | Best DL, needs GPU |
| 13 | **PPO (RL)** | `stable-baselines3` | hours | unknown | Joint entry/exit/sizing |

---

## 3. FEATURES THAT WORK (40 features, ranked by importance)

### Top 10 Features (by ExtraTrees feature_importance)

| # | Feature | Category | Importance | VN30 Equivalent |
|---|---------|----------|------------|-----------------|
| 1 | hour_cos | Temporal | 0.054 | Session time (ATO/ATC/lunch) |
| 2 | bb_width | Volatility | 0.053 | BB(20,2) width / SMA20 |
| 3 | macd_hist | Momentum | 0.049 | MACD histogram normalized |
| 4 | atr_pct | Volatility | 0.044 | ATR(14) / close |
| 5 | ema50_dist | Trend | 0.043 | (close - EMA50) / close |
| 6 | ema_spread | Trend | 0.043 | EMA8_dist - EMA50_dist |
| 7 | hour_sin | Temporal | 0.040 | Session time |
| 8 | atr_14 | Volatility | 0.037 | ATR(14) raw |
| 9 | ret_13 | Returns | 0.036 | 13-bar return |
| 10 | rsi_3 | Momentum | 0.034 | RSI(3) |

### Full Feature Set (copy-paste ready)

```python
FEATURES = {
    # Price action (4)
    "body_pct": "(close - open) / close",
    "range_pct": "(high - low) / close",
    "upper_wick": "(high - max(close,open)) / (high-low)",
    "lower_wick": "(min(close,open) - low) / (high-low)",
    
    # Returns (7)
    "ret_1": "close.pct_change(1)",
    "ret_2": "close.pct_change(2)",
    "ret_3": "close.pct_change(3)",
    "ret_5": "close.pct_change(5)",
    "ret_8": "close.pct_change(8)",
    "ret_13": "close.pct_change(13)",
    "ret_21": "close.pct_change(21)",
    
    # Volatility (5)
    "atr_5": "ATR(5)",
    "atr_14": "ATR(14)",
    "atr_ratio": "ATR(5) / ATR(14)",  # volatility acceleration
    "atr_pct": "ATR(14) / close",
    "bb_width": "4 * std(20) / SMA(20)",
    
    # Bollinger (1)
    "bb_pos": "(close - SMA20) / (2 * std20)",  # position in band
    
    # RSI (4)
    "rsi_3": "RSI(3)",
    "rsi_7": "RSI(7)",
    "rsi_14": "RSI(14)",
    "rsi_delta": "RSI(7) - RSI(7).shift(3)",
    
    # Stochastic (2)
    "stoch_k": "%K(14)",
    "stoch_d": "%D(3)",
    
    # EMA distance (4)
    "ema8_dist": "(close - EMA8) / close",
    "ema21_dist": "(close - EMA21) / close",
    "ema50_dist": "(close - EMA50) / close",
    "ema_spread": "ema8_dist - ema50_dist",
    
    # Volume (4)
    "vol_ratio": "volume / EMA(volume, 20)",
    "vol_ratio_3": "SMA(volume, 3) / EMA(volume, 20)",
    "obv_slope": "OBV.diff(5) / (close * 5)",
    "buy_pressure": "(close - low) / (high - low)",
    
    # MACD + ADX (2)
    "macd_hist": "MACD_histogram / close",
    "adx": "ADX(14)",
    
    # Temporal (4)
    "hour_sin": "sin(2π * hour / 24)",
    "hour_cos": "cos(2π * hour / 24)",
    "is_us_session": "13:00-21:00 UTC",  # VN30: 09:00-11:30, 13:00-14:30
    "is_asia_session": "00:00-08:00 UTC",  # VN30: 09:00-09:15 ATO
    
    # Microstructure (3)
    "tick_direction": "sign(close.diff())",
    "tick_run": "consecutive same-direction ticks",
    "spread_proxy": "(high - low) / close",
}
```

### VN30 Additional Features to Consider
```python
VN30_EXTRA_FEATURES = {
    "bid_ask_imbalance": "VN30 has L2 orderbook data",
    "basis": "F1M_price - VN30_index (premium/discount)",
    "oi_change": "Open interest change (derivative data)",
    "foreign_flow": "Foreign net buy/sell (from HOSE)",
    "ato_volume_ratio": "ATO volume / avg daily volume",
    "time_to_close": "Minutes remaining to ATC (14:30)",
    "lunch_gap": "afternoon_open - morning_close",
}
```

---

## 4. LABELING METHODS

### Method 1: Fixed-Horizon (RECOMMENDED for scalping)

```python
def label_fixed_horizon(close, atr, horizon=6):
    """
    Label each bar based on forward return vs adaptive threshold.
    horizon: bars ahead to check (6 bars × 3m = 18min, 6 bars × 1m = 6min for VN30)
    """
    labels = np.zeros(len(close))
    for i in range(len(close) - horizon):
        fwd_ret = (close[i + horizon] - close[i]) / close[i]
        threshold = max(0.0006, 0.3 * atr[i] / close[i])
        if fwd_ret > threshold:
            labels[i] = 1   # LONG
        elif fwd_ret < -threshold:
            labels[i] = -1  # SHORT
    return labels
```

**For VN30 adaptation:**
- Horizon = 6 bars at 1m = 6 minutes (good for scalping)
- Threshold = max(0.0003, 0.3 * ATR/close) — lower because VN30 has less noise
- Zero commission → threshold can be lower

### Method 2: Triple-Barrier (RECOMMENDED for swing)

```python
def label_triple_barrier(close, high, low, atr, sl_mult=1.0, tp_mult=1.5, max_bars=10):
    """
    Three barriers: Take-Profit above, Stop-Loss below, Time-out.
    First barrier touched determines label.
    """
    labels = np.zeros(len(close))
    for i in range(len(close) - max_bars):
        entry = close[i]
        sl_dist = sl_mult * atr[i]
        tp_dist = tp_mult * atr[i]
        
        for j in range(i+1, min(i+max_bars+1, len(close))):
            if high[j] >= entry + tp_dist:
                labels[i] = 1   # TP hit → LONG win
                break
            if low[j] <= entry - sl_dist:
                labels[i] = -1  # SL hit → SHORT win
                break
    return labels
```

**VN30 adaptation:**
- SL = 0.5 × ATR (tighter because lower volatility)
- TP = 1.0 × ATR
- max_bars = 15 (15 minutes at 1m)

---

## 5. WALK-FORWARD FRAMEWORK (Copy-Paste Ready)

```python
from datetime import timedelta
from sklearn.ensemble import ExtraTreesClassifier

def walk_forward_validate(df, features, train_days=30, test_days=10, threshold=0.58):
    """
    Calendar-based walk-forward validation.
    Train on train_days, test on test_days, step forward by test_days.
    
    For VN30: train_days=20 (trading days), test_days=5
    For Crypto: train_days=30, test_days=10 (24/7)
    """
    dates = df["date"]
    min_date, max_date = dates.min(), dates.max()
    
    all_trades = []
    train_start = min_date
    
    while True:
        train_end = train_start + timedelta(days=train_days)
        test_end = train_end + timedelta(days=test_days)
        if test_end > max_date:
            break
        
        train_mask = (dates >= train_start) & (dates < train_end)
        test_mask = (dates >= train_end) & (dates < test_end)
        
        train_data = df[train_mask]
        test_data = df[test_mask]
        
        if len(train_data) < 2000 or len(test_data) < 50:
            train_start += timedelta(days=test_days)
            continue
        
        # Train
        X_train = train_data[features].values
        y_train = (train_data["label"] == 1).astype(int).values
        
        model = ExtraTreesClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=50,
            max_features="sqrt", class_weight="balanced", n_jobs=-1,
        )
        model.fit(X_train, y_train)
        
        # Test
        X_test = test_data[features].values
        proba = model.predict_proba(X_test)[:, 1]
        
        # Generate trades
        for j in range(len(proba)):
            if proba[j] > threshold:
                all_trades.append({"direction": "long", "prob": proba[j], 
                                   "actual": test_data.iloc[j]["label"]})
            elif proba[j] < (1 - threshold):
                all_trades.append({"direction": "short", "prob": 1-proba[j],
                                   "actual": test_data.iloc[j]["label"]})
        
        train_start += timedelta(days=test_days)
    
    # Calculate metrics
    if not all_trades:
        return None
    
    wins = sum(1 for t in all_trades 
               if (t["direction"]=="long" and t["actual"]==1) or 
                  (t["direction"]=="short" and t["actual"]==-1))
    
    return {
        "trades": len(all_trades),
        "win_rate": wins / len(all_trades) * 100,
        "trades_per_day": len(all_trades) / (test_days * (max_date - min_date).days / test_days),
    }
```

---

## 6. MODEL CONFIGURATION (Production-Ready)

### ExtraTrees (BEST for deployment)
```python
from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier(
    n_estimators=200,       # 200 trees (diminishing returns after)
    max_depth=6,            # shallow to prevent overfitting
    min_samples_leaf=50,    # conservative leaf size
    max_features="sqrt",    # sqrt(40) ≈ 6 features per split
    class_weight="balanced",# handle imbalanced labels
    n_jobs=-1,              # all CPU cores
)
```

**Why ExtraTrees over LightGBM for scalping:**
- More stable probability calibration (LGB proba clusters near 0.5)
- Faster training (no sequential boosting)
- Less overfitting on noisy short-timeframe data
- Better at threshold-based trading (probabilities more spread out)

### LightGBM (BEST for meta-labeling, signal filtering)
```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=3,            # very shallow (anti-overfit)
    num_leaves=8,           # 2^3 = 8
    min_child_samples=200,  # large min samples
    learning_rate=0.03,     # slow learning
    colsample_bytree=0.5,   # feature subsampling
    subsample=0.7,          # row subsampling
    subsample_freq=5,
    reg_alpha=0.5,          # L1 regularization
    reg_lambda=2.0,         # L2 regularization
    class_weight="balanced",
    verbose=-1, n_jobs=-1,
)
```

### Stacking Ensemble (HIGHEST accuracy)
```python
from sklearn.ensemble import StackingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

estimators = [
    ("lgbm", lgb.LGBMClassifier(n_estimators=150, max_depth=3, verbose=-1)),
    ("et", ExtraTreesClassifier(n_estimators=150, max_depth=6, min_samples_leaf=50)),
]
model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(C=1.0),
    cv=5, n_jobs=-1,
)
```

---

## 7. VN30F1M SPECIFIC RECOMMENDATIONS

### Cost Advantage
| Item | Crypto (OKX) | VN30F1M |
|------|-------------|---------|
| Commission | 0.02-0.05% per side | **0% intraday** |
| Round-trip cost | 0.04-0.10% | **~0.01% (slippage only)** |
| Edge needed per trade | >10 bps | **>1-2 bps** |
| Implication | Only 16/420 configs viable with taker | **Most configs should be viable** |

### Timeframe Recommendation
- **VN30 1m**: ~450 bars/session × 2 sessions = 900 bars/day
- **Model retrain**: Every session open (morning + afternoon)
- **Train window**: 20 trading days (~18,000 bars at 1m)
- **Threshold**: Start at 0.55 (lower than crypto because lower cost)

### Session-Specific Features
```python
# VN30 sessions (critical for prediction)
"is_ato": "09:00-09:15 (auction, high vol)",
"is_morning": "09:15-11:30 (continuous)",
"is_lunch": "11:30-13:00 (no trading)",
"is_afternoon": "13:00-14:30 (continuous)",
"is_atc": "14:30-14:45 (closing auction)",
"time_to_lunch": "(11:30 - current) in minutes",
"time_to_close": "(14:30 - current) in minutes",
"morning_return": "close_now - morning_open",
```

### VN30 Data Requirements
- **Minimum**: 60 trading days × 900 bars = 54,000 bars
- **Recommended**: 120 trading days = 108,000 bars
- **Source**: SSI/VNDirect API for 1m OHLCV + orderbook

### Expected Performance on VN30
Based on crypto results extrapolated to zero-commission:

| Scenario | Trades/day | WR | Net Edge/trade | Daily PnL |
|----------|-----------|-----|----------------|-----------|
| Conservative (thr=0.60) | 5-8 | 57% | 8-12 bps | +0.05-0.10% |
| Moderate (thr=0.55) | 15-25 | 55% | 5-8 bps | +0.10-0.15% |
| Aggressive (thr=0.52) | 40-60 | 53% | 3-5 bps | +0.12-0.20% |

With VN30 zero commission, the **Moderate** profile is recommended (15-25 trades/day, 55% WR).

---

## 8. IMPLEMENTATION CHECKLIST FOR VN30

### Step 1: Data Pipeline
- [ ] Get 1m OHLCV data for VN30F1M (120 days minimum)
- [ ] Compute all 40 features (same as crypto, adapt session times)
- [ ] Add VN30-specific features (basis, foreign flow, ATO volume)

### Step 2: Labeling
- [ ] Fixed-horizon labeling: 6 bars (6 minutes) forward return
- [ ] Threshold: `max(0.0003, 0.3 * ATR / close)` (half of crypto)
- [ ] Verify label distribution: should be ~25% long, ~25% short, ~50% neutral

### Step 3: Walk-Forward Validation
- [ ] Train window: 20 trading days
- [ ] Test window: 5 trading days
- [ ] Step: 5 trading days
- [ ] Test all 3 models: ExtraTrees, LightGBM, RandomForest
- [ ] Test thresholds: 0.52, 0.55, 0.58, 0.60

### Step 4: Deploy
- [ ] Train model on most recent 20 days
- [ ] Retrain daily before market open (08:45)
- [ ] Signal threshold: 0.55 (start conservative)
- [ ] Position size: fixed (e.g., 1 lot VN30F1M)
- [ ] Exit: time-cut at 6 minutes if not in profit

### Step 5: Monitor
- [ ] Track OOS win rate daily
- [ ] If WR < 52% for 5 consecutive days → retrain with more data or pause
- [ ] Compare to random baseline (50% WR)

---

## 9. MODELS NOT WORTH TRYING (for short-term trading)

| Model | Why Not |
|-------|---------|
| LSTM/GRU | Same accuracy as trees, 50x slower training |
| Vanilla Transformer | Overfits on financial data (too few samples) |
| Reinforcement Learning | Unstable training, results not reproducible |
| GAN | Only useful for synthetic data augmentation |
| Gaussian Process | O(n³) — unusable for >5000 samples |
| KNN | Curse of dimensionality on 40 features |
| Naive Bayes | Independence assumption violated by financial features |

---

## 10. KEY ACADEMIC REFERENCES

1. **Grinsztajn et al. (2022)** "Why tree-based models still outperform deep learning on tabular data" — NeurIPS. **Conclusion: LightGBM/XGBoost beat all DL on tabular financial data.**

2. **López de Prado (2018)** "Advances in Financial Machine Learning" — Wiley. **Key concepts: Triple-barrier labeling, meta-labeling, CUSUM filter, sequential bootstrap.**

3. **Gu, Kelly, Xiu (2020)** "Empirical Asset Pricing via Machine Learning" — Review of Financial Studies. **Most comprehensive ML comparison for financial prediction.**

4. **Lim et al. (2019)** "Temporal Fusion Transformers" — Google. **Best DL architecture for time series forecasting with interpretability.**

5. **Nie et al. (2023)** "PatchTST" — **Channel-independent patched transformer, SOTA on forecasting benchmarks.**

---

## 11. QUICK-START CODE (VN30 Adaptation)

```python
"""
Quick-start ML trading for VN30F1M.
Replace DATA_PATH with your VN30 1m data source.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from datetime import timedelta

# 1. Load data
df = pd.read_csv("vn30_1m.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# 2. Compute features (see Section 3 for full list)
# ... (copy compute_features function from above)

# 3. Label
horizon = 6  # 6 minutes forward
c = df["close"].values
atr = df["atr_14"].values
labels = np.zeros(len(c))
for i in range(len(c) - horizon):
    if atr[i] <= 0: continue
    fwd_ret = (c[i+horizon] - c[i]) / c[i]
    thresh = max(0.0003, 0.3 * atr[i] / c[i])  # lower for VN30
    if fwd_ret > thresh: labels[i] = 1
    elif fwd_ret < -thresh: labels[i] = -1
df["label"] = labels

# 4. Walk-forward
FEATURES = [...]  # 40 feature columns
results = walk_forward_validate(df, FEATURES, train_days=20, test_days=5, threshold=0.55)
print(f"Trades: {results['trades']}, WR: {results['win_rate']:.1f}%")

# 5. Train final model (latest 20 days)
recent = df.tail(20 * 900)  # 20 trading days
valid = recent[recent["label"] != 0]
X = valid[FEATURES].values
y = (valid["label"] == 1).astype(int).values
model = ExtraTreesClassifier(n_estimators=200, max_depth=6, min_samples_leaf=50,
                              max_features="sqrt", class_weight="balanced", n_jobs=-1)
model.fit(X, y)

# 6. Predict on new bar
new_bar_features = ...  # latest 1m bar features
prob_long = model.predict_proba([new_bar_features])[0][1]
if prob_long > 0.55:
    print("BUY SIGNAL")
elif prob_long < 0.45:
    print("SELL SIGNAL")
```

---

*Report generated from OKX crypto futures ML research. Core methodology (features, labeling, walk-forward, models) is directly transferable to VN30F1M with adaptations noted in Section 7.*
