---
description: "Run VN30F1M signal scanner and report current market signals. Shows regime, active combos, and any triggered entries."
argument-hint: "[combo_name]"
---

# Signal Scanner

Scan VN30F1M for current trading signals.

## Arguments

```
$ARGUMENTS
```

**Argument interpretation**:
- If a combo name is provided (e.g., `D`, `J`, `X`): scan only that combo
- If empty: scan all active combos per regime filter

## Execution Procedure

1. **Check session** — verify within trading hours (9:00-11:30 or 13:00-14:30 Vietnam time)
2. **Read skill** — load `skills/signal-scanner/SKILL.md` for context
3. **Detect regime** — run `detect_regime()` from scanner.py on 15m data
4. **Check signal tracker** — identify disabled combos
5. **Scan signals** — for each active combo, evaluate current conditions
6. **Report** — output signals with entry/SL/TP/confidence

## Output

Report must include:
- Current time and session status
- Market regime (TRENDING/RANGING/VOLATILE/NORMAL)
- ATR value and ratio to SMA50
- For each signal: combo, direction, entry, SL, TP, confidence score
- Daily PnL status and remaining loss budget
