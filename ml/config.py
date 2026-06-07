"""ML configuration — loaded from strategy_config.yaml ml_filter section."""

DEFAULT_ML_CONFIG = {
    "enabled": False,
    "shadow_mode": True,
    "threshold": 0.45,
    "model_path": "ml/models/meta_label_latest.pkl",
    "retrain_days": 30,
    "kill_switch_days": 10,
}
