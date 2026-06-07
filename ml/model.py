"""
MLFilter: Production inference class for scanner integration.

Loads trained model and provides predict_proba interface.
Supports shadow mode (log-only, no veto) and kill-switch auto-disable.
"""
import pickle
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from ml.features import compute_indicators, extract_features_at, META_LABEL_FEATURES
from ml.config import DEFAULT_ML_CONFIG


class MLFilter:
    """ML meta-labeling filter for CB signals."""

    def __init__(self, config: dict = None):
        cfg = {**DEFAULT_ML_CONFIG, **(config or {})}
        self.enabled = cfg["enabled"]
        self.shadow_mode = cfg["shadow_mode"]
        self.threshold = cfg["threshold"]
        self.model_path = Path(cfg["model_path"])
        self.kill_switch_days = cfg["kill_switch_days"]

        self._model = None
        self._features = META_LABEL_FEATURES
        self._model_meta = {}
        self._shadow_log = []
        self._consecutive_underperform = 0

        if self.enabled:
            self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            print(f"[ML] Model not found: {self.model_path}")
            self.enabled = False
            return

        with open(self.model_path, "rb") as f:
            data = pickle.load(f)

        self._model = data["model"]
        self._features = data.get("features", META_LABEL_FEATURES)
        self._model_meta = {
            "trained_at": data.get("trained_at", "unknown"),
            "n_samples": data.get("n_samples", 0),
            "baseline_wr": data.get("baseline_wr", 0),
            "threshold": data.get("threshold", self.threshold),
        }
        self.threshold = data.get("threshold", self.threshold)
        print(f"[ML] Model loaded: {self.model_path}")
        print(f"     Trained: {self._model_meta['trained_at']} | "
              f"Samples: {self._model_meta['n_samples']} | "
              f"Threshold: {self.threshold:.2f}")

    def predict(self, df_5m: pd.DataFrame, signal_idx: int, direction: int) -> dict:
        """Predict P(win) for a CB signal.

        Args:
            df_5m: 5m OHLCV DataFrame with at least 50 bars.
            signal_idx: iloc position of the signal bar.
            direction: 1 for BUY, -1 for SELL.

        Returns:
            dict with: p_win, should_enter, reason
        """
        if not self.enabled or self._model is None:
            return {"p_win": 1.0, "should_enter": True, "reason": "ML_DISABLED"}

        # Compute indicators if not already done
        if "atr_14" not in df_5m.columns:
            df_5m = compute_indicators(df_5m)

        features = extract_features_at(df_5m, signal_idx, direction)
        if features is None:
            return {"p_win": 1.0, "should_enter": True, "reason": "FEATURE_ERROR"}

        # Build feature vector in correct order
        x = np.array([[features.get(f, 0.0) for f in self._features]])
        x = np.nan_to_num(x, nan=0.0)

        p_win = float(self._model.predict_proba(x)[0][1])
        should_enter = p_win >= self.threshold

        result = {
            "p_win": p_win,
            "should_enter": should_enter,
            "reason": "ML_PASS" if should_enter else "ML_VETO",
        }

        if self.shadow_mode:
            result["should_enter"] = True
            result["reason"] = f"SHADOW(p={p_win:.3f},{'PASS' if p_win >= self.threshold else 'VETO'})"
            self._shadow_log.append({
                "timestamp": datetime.now().isoformat(),
                "p_win": p_win,
                "threshold": self.threshold,
                "would_veto": not should_enter,
                "direction": "BUY" if direction == 1 else "SELL",
            })

        return result

    def log_trade_result(self, p_win: float, actual_pnl: float):
        """Track trade results for kill-switch evaluation."""
        self._shadow_log.append({
            "timestamp": datetime.now().isoformat(),
            "p_win": p_win,
            "actual_pnl": actual_pnl,
            "type": "result",
        })

    def get_shadow_report(self) -> dict:
        """Generate shadow mode report."""
        if not self._shadow_log:
            return {"status": "no_data"}

        predictions = [e for e in self._shadow_log if "would_veto" in e]
        results = [e for e in self._shadow_log if e.get("type") == "result"]

        report = {
            "total_signals": len(predictions),
            "would_veto": sum(1 for p in predictions if p["would_veto"]),
            "retention_rate": (1 - sum(1 for p in predictions if p["would_veto"]) / max(1, len(predictions))) * 100,
            "results_tracked": len(results),
        }

        if results:
            vetoed_results = [r for r, p in zip(results, predictions)
                           if p.get("would_veto") and r.get("actual_pnl") is not None]
            if vetoed_results:
                veto_correct = sum(1 for r in vetoed_results if r["actual_pnl"] <= 0)
                report["veto_precision"] = veto_correct / len(vetoed_results) * 100

        return report

    def save_shadow_log(self, path="ml/reports/shadow_log.json"):
        """Persist shadow log for analysis."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._shadow_log, f, indent=2)

    @property
    def is_active(self) -> bool:
        return self.enabled and self._model is not None
