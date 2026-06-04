---
description: "Detect current VN30F1M intraday market regime (TRENDING/RANGING/VOLATILE/NORMAL) with supporting metrics."
argument-hint: ""
---

# Regime Detector

Check current market regime for VN30F1M.

## Execution Procedure

1. **Read skill** — load `skills/regime-detector/SKILL.md`
2. **Fetch data** — get latest 50 bars of 15m VN30F1M data
3. **Compute indicators** — ADX(14), DI+, DI-, ATR(14), ATR_SMA50
4. **Classify** — apply regime rules from detect_regime()
5. **Report** — output regime with all supporting metrics

## Output

Must include:
- Regime classification with confidence level
- ADX value and DI+/DI- spread
- ATR current vs SMA50 (ratio)
- Which combo categories are active/disabled
- Recommendation: trade normally / reduce size / skip
