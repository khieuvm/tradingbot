"""
Standalone ML Signal Generator for scanner integration.

Generates BUY/SELL signals independently of CB, based on direction prediction.
Best config: Horizon=6, threshold=0.55, AM+SELL filter.

Usage in scanner:
    generator = StandaloneMLSignal(config)
    signal = generator.check_signal(df_5m)
    if signal: # {"direction": -1, "prob": 0.62, ...}
"""
import pickle
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from ml.standalone_signals import compute_all_features, FEATURE_COLS, COST


MODEL_PATH = Path("ml/models/standalone_ml_latest.pkl")
SHADOW_LOG_PATH = Path("ml/reports/standalone_shadow_log.json")

DEFAULT_STANDALONE_CONFIG = {
    "enabled": False,
    "shadow_mode": True,
    "threshold": 0.55,
    "horizon": 6,
    "session_filter": ["AM"],
    "direction_filter": ["SELL"],
    "max_trades_per_session": 2,
    "model_path": str(MODEL_PATH),
}


class StandaloneMLSignal:
    """Standalone ML signal generator — independent of CB."""

    def __init__(self, config: dict = None):
        cfg = {**DEFAULT_STANDALONE_CONFIG, **(config or {})}
        self.enabled = cfg["enabled"]
        self.shadow_mode = cfg["shadow_mode"]
        self.threshold = cfg["threshold"]
        self.horizon = cfg["horizon"]
        self.session_filter = cfg["session_filter"]
        self.direction_filter = cfg["direction_filter"]
        self.max_trades_per_session = cfg["max_trades_per_session"]
        self.model_path = Path(cfg["model_path"])

        self._model = None
        self._features = FEATURE_COLS
        self._shadow_log = []
        self._session_trades = {}
        self._last_signal_bar = -999

        if self.enabled:
            self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            print(f"[ML-Standalone] Model not found: {self.model_path}")
            self.enabled = False
            return

        with open(self.model_path, "rb") as f:
            data = pickle.load(f)

        self._model = data["model"]
        self._features = data.get("features", FEATURE_COLS)
        meta = data.get("meta", {})
        print(f"[ML-Standalone] Model loaded: {self.model_path}")
        print(f"     Config: H={self.horizon}, thr={self.threshold}, "
              f"session={self.session_filter}, dir={self.direction_filter}")
        print(f"     Trained: {meta.get('trained_at', '?')} | "
              f"OOS: WR={meta.get('oos_wr', '?')}%, PF={meta.get('oos_pf', '?')}")

    def check_signal(self, df_5m: pd.DataFrame) -> dict | None:
        """Check if current bar generates a standalone ML signal.

        Args:
            df_5m: 5m OHLCV DataFrame (at least 60 bars for indicator warmup).

        Returns:
            dict with signal info, or None if no signal.
            {"direction": 1/-1, "prob": float, "action": "ENTER"/"SHADOW", ...}
        """
        if not self.enabled or self._model is None:
            return None

        if len(df_5m) < 60:
            return None

        # Ensure time columns exist (scanner df may not have them)
        df_5m = df_5m.copy()
        if "time" in df_5m.columns:
            df_5m["time"] = pd.to_datetime(df_5m["time"])
        if "mins" not in df_5m.columns and "time" in df_5m.columns:
            df_5m["mins"] = df_5m["time"].dt.hour * 60 + df_5m["time"].dt.minute
        if "session" not in df_5m.columns and "mins" in df_5m.columns:
            df_5m["session"] = np.where(df_5m["mins"] < 12*60, "AM", "PM")
        if "date" not in df_5m.columns and "time" in df_5m.columns:
            df_5m["date"] = df_5m["time"].dt.date

        # Compute features on the full dataframe
        if "atr_14" not in df_5m.columns or "ema_align" not in df_5m.columns:
            df_5m = compute_all_features(df_5m)

        idx = len(df_5m) - 1
        row = df_5m.iloc[idx]

        # Time check
        mins = int(row.get("mins", 0))
        session = row.get("session", "AM")

        if session == "AM" and mins >= 11*60+20:
            return None
        if session == "PM" and mins >= 14*60+20:
            return None

        # Session filter
        if self.session_filter and session not in self.session_filter:
            return None

        # Cooldown (horizon bars between signals)
        bar_time = pd.to_datetime(row["time"]) if "time" in row.index else None
        if idx - self._last_signal_bar < self.horizon:
            return None

        # Session trade count
        date_val = row.get("date", "")
        session_key = (str(date_val), session)
        if self._session_trades.get(session_key, 0) >= self.max_trades_per_session:
            return None

        # Build feature vector
        available = [f for f in self._features if f in df_5m.columns]
        x = np.array([[float(row.get(f, 0.0)) if not pd.isna(row.get(f, 0.0)) else 0.0
                       for f in available]])
        x = np.nan_to_num(x, nan=0.0)

        # Predict
        prob = float(self._model.predict_proba(x)[0][1])

        # Determine direction
        direction = 0
        if prob > self.threshold:
            direction = 1
        elif prob < (1 - self.threshold):
            direction = -1

        if direction == 0:
            return None

        # Direction filter
        dir_str = "BUY" if direction == 1 else "SELL"
        if self.direction_filter and dir_str not in self.direction_filter:
            return None

        # Signal generated
        self._last_signal_bar = idx
        self._session_trades[session_key] = self._session_trades.get(session_key, 0) + 1

        signal = {
            "direction": direction,
            "direction_str": dir_str,
            "prob": prob,
            "session": session,
            "time": str(row.get("time", "")),
            "price": float(row["close"]),
            "atr": float(row.get("atr_14", 3.0)),
            "horizon": self.horizon,
        }

        if self.shadow_mode:
            signal["action"] = "SHADOW"
            self._shadow_log.append({
                "timestamp": datetime.now().isoformat(),
                **signal,
            })
            return signal
        else:
            signal["action"] = "ENTER"
            self._shadow_log.append({
                "timestamp": datetime.now().isoformat(),
                **signal,
            })
            return signal

    def log_result(self, pnl: float, signal_time: str):
        """Log trade result for performance tracking."""
        self._shadow_log.append({
            "timestamp": datetime.now().isoformat(),
            "type": "result",
            "pnl": pnl,
            "signal_time": signal_time,
        })

    def save_log(self):
        """Save shadow log to disk."""
        SHADOW_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SHADOW_LOG_PATH, "w") as f:
            json.dump(self._shadow_log, f, indent=2)

    def get_report(self) -> dict:
        """Summary report of shadow signals."""
        signals = [s for s in self._shadow_log if "direction" in s]
        results = [s for s in self._shadow_log if s.get("type") == "result"]
        return {
            "total_signals": len(signals),
            "buy_signals": sum(1 for s in signals if s.get("direction") == 1),
            "sell_signals": sum(1 for s in signals if s.get("direction") == -1),
            "results_tracked": len(results),
            "total_pnl": sum(r["pnl"] for r in results) if results else 0,
        }

    @property
    def is_active(self) -> bool:
        return self.enabled and self._model is not None


def train_and_save_model(days: int = 180):
    """Train standalone ML model and save for production use.

    This retrains using ALL available data (no walk-forward split needed
    since we're deploying, not validating).
    """
    from backtest.engine import load
    from ml.standalone_signals import label_fixed_horizon

    print("[ML-Standalone] Training production model...")
    df = load("5m", days=days)
    df = compute_all_features(df)
    df["label"] = label_fixed_horizon(df, horizon=6)

    available = [f for f in FEATURE_COLS if f in df.columns]
    X = df[available].values
    y = df["label"].values

    # Train on labeled bars only
    labeled_mask = y != 0
    X_train = np.nan_to_num(X[labeled_mask], nan=0.0)
    y_train = (y[labeled_mask] == 1).astype(int)

    print(f"  Training samples: {len(X_train)} (LONG={y_train.sum()}, SHORT={len(y_train)-y_train.sum()})")

    model = ExtraTreesClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=50,
        max_features="sqrt", class_weight="balanced", n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Save
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "model": model,
        "features": available,
        "meta": {
            "trained_at": datetime.now().isoformat(),
            "n_samples": len(X_train),
            "n_long": int(y_train.sum()),
            "n_short": int(len(y_train) - y_train.sum()),
            "horizon": 6,
            "threshold": 0.55,
            "n_features": len(available),
            "oos_wr": 59.7,
            "oos_pf": 1.57,
            "config": "AM+SELL, H=6, thr=0.55",
        },
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(data, f)

    print(f"  Model saved: {MODEL_PATH}")
    print(f"  Features: {len(available)}")
    return model


if __name__ == "__main__":
    train_and_save_model()
