# VN30F1M Trading Workflows

## Daily Trading Workflow

**Cadence:** Every trading day (Mon-Fri), automated sequence
**Sessions:** 9:00-11:30 (AM) and 13:00-14:30 (PM), UTC+7

### Pre-Market (8:45-9:00)
1. `regime-detector` — compute initial ADX/ATR from previous session data
2. `signal-scanner` — verify scanner is running, check config
3. Review previous day's `trade-postmortem` action items
4. Check: any combos auto-disabled by signal tracker? Why?

### Morning Session (9:00-11:30)
| Time | Action |
|------|--------|
| 9:00-9:15 | Scanner running but NO entries (regime forming) |
| 9:15 | Regime confirmed → begin generating signals |
| 9:15-10:45 | **AM PRIME** — full confidence entries |
| 10:45-11:15 | Reduce new entry confidence (half size) |
| 11:15-11:30 | NO new entries. Close positions or set tight trail |
| 11:30 | Market closed. Scanner pauses. |

### Lunch Break (11:30-13:00)
- Scanner OFF (market closed)
- Review AM session performance
- Prepare PM regime assessment

### Afternoon Session (13:00-14:30)
| Time | Action |
|------|--------|
| 13:00 | Market reopens. Re-detect regime (fresh assessment) |
| 13:00-13:15 | Scanner resumes, regime forming |
| 13:15-14:00 | **PM PRIME** — signals active (lower vol than AM) |
| 14:00-14:15 | Last entries allowed (HIGH confidence only) |
| 14:15-14:28 | **NO NEW ENTRIES** — flatten all positions |
| 14:28 | Force close ALL remaining positions |
| 14:30 | Market closed. Scanner OFF. |

### Post-Session (14:30-15:00)
1. `trade-postmortem` — classify all trades from today
2. Update signal tracker with outcomes
3. Check if any combo needs to be disabled
4. Log daily summary to `logs/daily_YYYY-MM-DD.log`

---

## Weekly Validation Workflow

**Cadence:** Every Sunday

### Steps
1. `trade-postmortem` (period=week) — full weekly review
2. Split analysis: AM vs PM performance (they differ!)
3. Identify combos with WR < 45% over the week
4. `backtest-validator` — re-validate any degrading combos
5. Update `strategy_config.yaml` grades if needed
6. Review: are SESSION_KILL trades > 20%? (entry timing issue)
7. Review regime detection accuracy (predicted vs actual outcomes)

---

## Monthly Edge Discovery Workflow

**Cadence:** First weekend of each month

### Steps
1. `trade-postmortem` (period=month) — full monthly review
2. Identify gaps: what types of moves are we missing?
3. Analyze: AM edge vs PM edge separately
4. `edge-researcher` — research 2-3 new hypotheses based on gaps
5. **Constraint:** all new edges must work within session time limits
6. `backtest-validator` — validate with intraday rules (14:28 close, lunch break)
7. If new edge grades A/B → add to scanner config
8. If existing combo degrades to F → disable and replace

---

## Combo Lifecycle

```
[edge-researcher] → discover hypothesis (must fit session time window)
       ↓
[backtest-validator] → walk-forward + MC (with 14:28 flatten, session rules)
       ↓
Grade A/B → [signal-scanner] → deploy to live scanning
       ↓
[trade-postmortem] → monitor rolling performance (AM/PM split)
       ↓
WR < 40% over 10 trades → auto-disable 24h
       ↓
Monthly re-validation → if still F → permanent disable
```

## Critical Intraday Rules (Apply to ALL Workflows)

1. **No overnight holds** — ALL positions close by 14:28, no exceptions
2. **Lunch break = dead time** — 11:30-13:00 market closed, no signals
3. **Session independence** — AM and PM are treated as separate trading days
4. **Time-aware entries** — don't enter if not enough time for TP
5. **14:15 cutoff** — last possible entry time
6. **Daily loss cap -10 pts** — stop all trading for the day if hit
7. **Cost = 0.96 pts/trade** (slippage 0.5 + commission 0.46) — NOT 1.74

---

## Research Workflow (Edge Improvement)

**Trigger:** Live anomaly, MFE pattern, or "why does X happen?"

### Steps
1. Quantify the problem precisely (e.g. "48/223 trades reverse at 4–6 pts MFE")
2. Write a focused research script (e.g. `research_mfe46.py`)
3. Identify the features that correlate with the problem
4. Write adaptive logic script (e.g. `research_adaptive_exit.py`)
5. Test on all relevant TFs with multi-threshold sweep (e.g. `bt_adaptive_3tf.py`)
6. Update `proven_edges.md` with findings (positive AND negative)
7. Update `anti_patterns.md` if finding is a confirmed dead end

### Research Scripts Convention
- `research_*.py` — exploratory: measure a specific phenomenon
- `bt_*.py` — backtest: test parameters, configurations, multi-TF
- `debug_*.py` — deep dive: individual TF analysis with trade detail

### Validation threshold for deploying a new rule
- Must improve P/D on the primary TF (5m, 129+ days)
- Delta PF > 0.2 OR delta P/D > 0.2/d
- No reduction in trade frequency (same signal count, different exits)
- Confirm rule makes logical sense (not curve-fitting)
