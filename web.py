"""
Trading Bot Web Dashboard
==========================
Run: streamlit run web.py

"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pandas_ta as ta
import json
from config import Config
from src.data_fetcher import DataFetcher
from src.strategy import MACrossStrategy, RSIStrategy, MACDStrategy
from src.backtest import Backtester
from src.signals import (
    COMBO_PRESETS, COND_LABELS, ALL_COND_KEYS,
    analyze_signal_performance, generate_combined_signals,
)
from src.notifier import TelegramNotifier
# --- TradingView color palette ---
TV_BG = "#131722"
TV_GRID = "#1e222d"
TV_TEXT = "#d1d4dc"
TV_UP = "#26a69a"
TV_DOWN = "#ef5350"
TV_UP_VOLUME = "rgba(38,166,154,0.5)"
TV_DOWN_VOLUME = "rgba(239,83,80,0.5)"
COLOR_BULL = 'rgba(38,166,154,0.9)'
COLOR_BEAR = 'rgba(239,83,80,0.9)'
TV_LAYOUT = dict(
    paper_bgcolor=TV_BG,
    plot_bgcolor=TV_BG,
    font=dict(color=TV_TEXT, family="Trebuchet MS, sans-serif"),
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=TV_TEXT),
    ),
    margin=dict(l=60, r=20, t=40, b=30),
)
TV_XAXIS = dict(gridcolor=TV_GRID, gridwidth=0.5, zeroline=False, showline=False, fixedrange=False)
TV_YAXIS = dict(gridcolor=TV_GRID, gridwidth=0.5, zeroline=False, showline=False, side="right", fixedrange=False)

def render_tv_chart(df_chart, title="", indicators=None, signals_list=None, chart_key="main"):

    """Render TradingView-quality chart using Apache ECharts.
    Drag = pan (custom handler), scroll = zoom (ECharts dataZoom inside).
    NO drag-to-zoom anywhere — ECharts doesn't have it by default.

    """

    def _j(s):

        """Series to JSON-safe list (NaN -> None)."""
        return [None if pd.isna(v) else round(float(v), 6) for v in s]
    dfc = df_chart.copy()
    if "time" in dfc.columns:
        dfc["_dt"] = pd.to_datetime(dfc["time"])
    else:
        dfc["_dt"] = pd.to_datetime(dfc.index)
    td = dfc["_dt"].diff().median() if len(dfc) > 1 else pd.Timedelta(days=1)
    intraday = td is not None and td < pd.Timedelta(days=1)
    fmt = "%m/%d %H:%M" if intraday else "%Y-%m-%d"
    dates = dfc["_dt"].dt.strftime(fmt).tolist()
    # ECharts candlestick format: [open, close, low, high]
    ohlc = dfc[["open", "close", "low", "high"]].fillna(0).values.tolist()
    vols = dfc["volume"].fillna(0).tolist()
    has_rsi = "RSI" in indicators
    has_macd = "MACD" in indicators
    n_sub = (1 if has_rsi else 0) + (1 if has_macd else 0)
    # --- Grid layout ---
    if n_sub == 0:
        grids = [{"left": 60, "right": 60, "top": 50, "bottom": 60}]
    elif n_sub == 1:
        grids = [
            {"left": 60, "right": 60, "top": 50, "bottom": "30%"},
            {"left": 60, "right": 60, "top": "74%", "bottom": 50},
        ]
    else:
        grids = [
            {"left": 60, "right": 60, "top": 50, "bottom": "42%"},
            {"left": 60, "right": 60, "top": "62%", "height": "14%"},
            {"left": 60, "right": 60, "top": "79%", "bottom": 50},
        ]
    ax_line = {"lineStyle": {"color": "#485c7b"}}
    lb_style = {"color": "#848e9c", "fontSize": 11}
    spl = {"lineStyle": {"color": "#1e222d", "type": "dashed"}}
    x_axes = [{"type": "category", "data": dates, "gridIndex": 0, "axisLine": ax_line,
               "axisLabel": {"show": n_sub == 0, **lb_style}, "axisTick": {"show": False}, "boundaryGap": True}]
    max_v = max(vols) if vols else 1
    y_axes = [
        {"scale": True, "gridIndex": 0, "position": "right", "splitLine": spl,
         "axisLabel": lb_style, "axisLine": {"show": False}},
        {"scale": True, "gridIndex": 0, "show": False, "max": max_v * 5},
    ]
    dz_xi = [0]
    gi, yi = 1, 2
    rsi_xi = rsi_yi = macd_xi = macd_yi = None
    if has_rsi:
        x_axes.append({"type": "category", "data": dates, "gridIndex": gi, "axisLine": ax_line,
                        "axisLabel": {"show": not has_macd, **lb_style}, "axisTick": {"show": False}})
        y_axes.append({"gridIndex": gi, "position": "right", "min": 0, "max": 100,
                        "splitLine": spl, "axisLabel": lb_style, "axisLine": {"show": False}})
        rsi_xi, rsi_yi = gi, yi; dz_xi.append(gi); gi += 1; yi += 1
    if has_macd:
        x_axes.append({"type": "category", "data": dates, "gridIndex": gi, "axisLine": ax_line,
                        "axisLabel": {**lb_style}, "axisTick": {"show": False}})
        y_axes.append({"scale": True, "gridIndex": gi, "position": "right",
                        "splitLine": spl, "axisLabel": lb_style, "axisLine": {"show": False}})
        macd_xi, macd_yi = gi, yi; dz_xi.append(gi); gi += 1; yi += 1
    # --- Series ---
    series = []
    ln = {"symbol": "none", "smooth": False}
    # Candlestick
    series.append({
        "type": "candlestick", "data": ohlc, "xAxisIndex": 0, "yAxisIndex": 0,
        "itemStyle": {"color": TV_UP, "color0": TV_DOWN, "borderColor": TV_UP, "borderColor0": TV_DOWN},
    })
    # Volume
    vd = [{"value": v, "itemStyle": {"color": TV_UP if c >= o else TV_DOWN, "opacity": 0.45}}
          for v, o, c in zip(vols, dfc["open"], dfc["close"])]
    series.append({"type": "bar", "data": vd, "xAxisIndex": 0, "yAxisIndex": 1, "barWidth": "60%"})
    # Overlays
    if "EMA 200" in indicators:
        series.append({**ln, "type": "line", "data": _j(ta.ema(dfc["close"], length=200)),
                       "xAxisIndex": 0, "yAxisIndex": 0, "lineStyle": {"color": "#f5c842", "width": 2}, "name": "EMA 200"})
    if "MA Cross" in indicators:
        series.append({**ln, "type": "line", "data": _j(ta.sma(dfc["close"], length=10)),
                       "xAxisIndex": 0, "yAxisIndex": 0, "lineStyle": {"color": "#2196F3", "width": 2}, "name": "SMA 10"})
        series.append({**ln, "type": "line", "data": _j(ta.sma(dfc["close"], length=20)),
                       "xAxisIndex": 0, "yAxisIndex": 0, "lineStyle": {"color": "#FF9800", "width": 2}, "name": "SMA 20"})
    if "Bollinger Bands" in indicators:
        bb = ta.bbands(dfc["close"], length=20, std=2)
        if bb is not None:
            series.append({**ln, "type": "line", "data": _j(bb.iloc[:, 0]), "xAxisIndex": 0, "yAxisIndex": 0,
                           "lineStyle": {"color": "rgba(33,150,243,0.6)", "width": 1}, "name": "BB Upper"})
            series.append({**ln, "type": "line", "data": _j(bb.iloc[:, 2]), "xAxisIndex": 0, "yAxisIndex": 0,
                           "lineStyle": {"color": "rgba(33,150,243,0.6)", "width": 1}, "name": "BB Lower"})
            series.append({**ln, "type": "line", "data": _j(bb.iloc[:, 1]), "xAxisIndex": 0, "yAxisIndex": 0,
                           "lineStyle": {"color": "rgba(33,150,243,0.4)", "width": 1, "type": "dashed"}, "name": "BB Mid"})
    # Signal markers
    if signals_list:
        dfc_ts = (dfc["_dt"].astype(np.int64) // 10**9).astype(int)
        ts_map = dict(zip(dfc_ts, range(len(dfc_ts))))
        for sl, cc, sd in signals_list:
            sig = sd.copy()
            if "time" in sig.columns:
                sig["_ts"] = (pd.to_datetime(sig["time"]).astype(np.int64) // 10**9).astype(int)
            else:
                sig["_ts"] = (pd.to_datetime(sig.index).astype(np.int64) // 10**9).astype(int)
            bp, sp_pts = [], []
            has_confidence = "signal_confidence" in sig.columns
            for _, r in sig[sig["signal"] == 1].iterrows():
                ix = ts_map.get(int(r["_ts"]))
                if ix is not None:
                    conf = int(r.get("signal_confidence", 1)) if has_confidence else 1
                    sz = {0: 10, 1: 10, 2: 14, 3: 18}.get(conf, 14)
                    bp.append({"value": [ix, float(r["low"]) * 0.997], "symbol": "triangle", "symbolSize": sz})
            for _, r in sig[sig["signal"] == -1].iterrows():
                ix = ts_map.get(int(r["_ts"]))
                if ix is not None:
                    conf = int(r.get("signal_confidence", 1)) if has_confidence else 1
                    sz = {0: 10, 1: 10, 2: 14, 3: 18}.get(conf, 14)
                    sp_pts.append({"value": [ix, float(r["high"]) * 1.003],
                                   "symbol": "triangle", "symbolSize": sz, "symbolRotate": 180})
            if bp:
                series.append({"type": "scatter", "data": bp, "xAxisIndex": 0, "yAxisIndex": 0,
                               "itemStyle": {"color": cc["buy"]}, "name": f"{sl} BUY", "z": 10})
            if sp_pts:
                series.append({"type": "scatter", "data": sp_pts, "xAxisIndex": 0, "yAxisIndex": 0,
                               "itemStyle": {"color": cc["sell"]}, "name": f"{sl} SELL", "z": 10})
    # RSI
    if has_rsi and rsi_xi is not None:
        rv = ta.rsi(dfc["close"], length=14)
        if rv is not None:
            series.append({**ln, "type": "line", "data": _j(rv), "xAxisIndex": rsi_xi, "yAxisIndex": rsi_yi,
                           "lineStyle": {"color": "#E040FB", "width": 2}, "name": "RSI",
                           "markLine": {"silent": True, "symbol": "none", "label": {"show": False},
                                        "data": [
                                            {"yAxis": 70, "lineStyle": {"color": "rgba(239,83,80,0.4)", "type": "dashed", "width": 1}},
                                            {"yAxis": 30, "lineStyle": {"color": "rgba(38,166,154,0.4)", "type": "dashed", "width": 1}},
                                        ]}})
    # MACD
    if has_macd and macd_xi is not None:
        mr = ta.macd(dfc["close"], fast=12, slow=26, signal=9)
        if mr is not None:
            mh = mr.iloc[:, 1]
            hd = [{"value": None if pd.isna(v) else round(float(v), 4),
                   "itemStyle": {"color": TV_UP if (not pd.isna(v) and v >= 0) else TV_DOWN}} for v in mh]
            series.append({"type": "bar", "data": hd, "xAxisIndex": macd_xi, "yAxisIndex": macd_yi,
                           "barWidth": "60%", "name": "Hist"})
            series.append({**ln, "type": "line", "data": _j(mr.iloc[:, 0]), "xAxisIndex": macd_xi, "yAxisIndex": macd_yi,
                           "lineStyle": {"color": "#2196F3", "width": 2}, "name": "MACD"})
            series.append({**ln, "type": "line", "data": _j(mr.iloc[:, 2]), "xAxisIndex": macd_xi, "yAxisIndex": macd_yi,
                           "lineStyle": {"color": "#FF9800", "width": 2}, "name": "Signal"})
    # DataZoom — scroll = zoom, NO drag-to-zoom (ECharts default)
    total = len(dates)
    vis = min(80, total)
    st_pct = max(0, (1 - vis / total) * 100)
    dz = [
        {"type": "inside", "xAxisIndex": dz_xi, "zoomOnMouseWheel": True,
         "moveOnMouseMove": False, "moveOnMouseWheel": False, "start": st_pct, "end": 100},
        {"type": "slider", "xAxisIndex": dz_xi, "bottom": 8, "height": 20,
         "borderColor": "#485c7b", "backgroundColor": "#1a1e2e",
         "fillerColor": "rgba(38,166,154,0.15)", "handleStyle": {"color": "#585f72"},
         "textStyle": {"color": "#848e9c"}, "start": st_pct, "end": 100,
         "dataBackground": {"lineStyle": {"color": "#485c7b"}, "areaStyle": {"color": "rgba(72,92,123,0.2)"}}},
    ]
    option = {
        "backgroundColor": TV_BG, "animation": False,
        "grid": grids, "xAxis": x_axes, "yAxis": y_axes,
        "series": series, "dataZoom": dz,
        "axisPointer": {"link": [{"xAxisIndex": "all"}], "lineStyle": {"color": "#485c7b", "type": "dashed"}},
        "tooltip": {"trigger": "axis",
                    "axisPointer": {"type": "cross", "crossStyle": {"color": "#485c7b"},
                                    "label": {"backgroundColor": "#1e222d", "color": "#d1d4dc"}},
                    "backgroundColor": "rgba(19,23,34,0.95)", "borderColor": "#485c7b",
                    "textStyle": {"color": "#d1d4dc", "fontSize": 12}},
        "title": {"text": title, "left": "center", "top": 8,
                  "textStyle": {"color": "#d1d4dc", "fontSize": 16, "fontWeight": "normal",
                                "fontFamily": "Trebuchet MS, sans-serif"}},
    }
    oj = json.dumps(option, ensure_ascii=False)
    h = 520 + (170 if has_rsi else 0) + (170 if has_macd else 0)
    cid = chart_key.replace(" ", "_")
    html = f"""<!DOCTYPE html>
<html><head>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:{TV_BG};overflow:hidden}}
#{cid}{{width:100%;height:{h}px}}
</style></head><body>
<div id="{cid}"></div>
<script>
(function(){{
var el=document.getElementById('{cid}');
var ch=echarts.init(el,null,{{renderer:'canvas'}});
ch.setOption({oj});
/* ── Drag-to-pan (custom: no zoom, only pan) ── */
var drag=false,sx=0;
ch.getZr().on('mousedown',function(p){{
  if(p.event.offsetY>el.clientHeight-50)return;
  drag=true;sx=p.event.clientX;
}});
ch.getZr().on('mousemove',function(p){{
  if(!drag)return;
  var x=p.event.clientX,dx=x-sx;sx=x;
  var o=ch.getOption().dataZoom[0];
  var r=o.end-o.start;
  var sh=-(dx/el.clientWidth)*r*1.5;
  var ns=Math.max(0,Math.min(100-r,o.start+sh));
  ch.dispatchAction({{type:'dataZoom',start:ns,end:ns+r}});
}});
ch.getZr().on('mouseup',function(){{drag=false}});
ch.getZr().on('globalout',function(){{drag=false}});
/* ── Responsive ── */
new ResizeObserver(function(){{ch.resize()}}).observe(el);
}})();
</script></body></html>"""
    import base64 as _b64
    _html_b64 = _b64.b64encode(html.encode("utf-8")).decode("utf-8")
    st.iframe(f"data:text/html;base64,{_html_b64}", height=h + 20)
# --- Page Config ---
st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="chart",
    layout="wide",
)
st.title("Trading Bot Dashboard")
# --- Sidebar (form with Apply button) ---
with st.sidebar.form("config_form"):
    st.header("Config")
    # Market type
    market_type = st.selectbox(
        "Market Type",
        ["Futures"],
        # ["Futures", "Index"],
    )
    # Symbol input based on market type
    if market_type == "Futures":
        symbol = "VN30F1M"
        st.text_input("Contract", value="VN30F1M", disabled=True)
    else:
        symbol = st.selectbox(
            "Index",
            ["VNINDEX", "VN30", "HNX30", "UPCOM"],
        )
    # Timeframe
    st.subheader("Time Frame")
    interval = st.select_slider(
        "Interval",
        options=["1m", "5m", "15m", "30m", "1H", "1D", "1W", "1M"],
        value="5m",
    )
    # Quick date range
    date_range_options = {
        "3 days": 3,
        "5 days": 5,
        "10 days": 10,
        "30 days": 30,
        "90 days": 90,
        "180 days": 180,
        "1 year": 365,
        "2 years": 730,
    }
    date_range = st.select_slider(
        "Period",
        options=list(date_range_options.keys()),
        value="90 days",
    )
    days_back = date_range_options[date_range]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    # Strategy parameters
    st.subheader("Strategy")
    st.caption("All strategies shown on chart")
    st.markdown("**MA Crossover**")
    fast_ma = st.slider("MA nhanh", 5, 50, 10)
    slow_ma = st.slider("Slow MA", 10, 200, 20)
    st.markdown("**RSI**")
    rsi_period = st.slider("RSI Period", 5, 30, 7)
    oversold = st.slider("Oversold", 10, 40, 35)
    overbought = st.slider("Overbought", 60, 90, 70)
    st.markdown("**MACD**")
    macd_fast = st.slider("MACD Fast", 5, 20, 12)
    macd_slow = st.slider("MACD Slow", 15, 50, 26)
    macd_signal = st.slider("MACD Signal", 5, 20, 9)
    st.markdown("**Volume Filter**")
    vol_mult = st.slider("RVOL (x average)", 1.0, 3.0, 1.5, step=0.1)
    # --- Signal conditions ---
    st.subheader("Entry Conditions")
    # Combo preset selector
    combo_names = list(COMBO_PRESETS.keys())
    selected_combo = st.selectbox(
        "Combo Preset",
        combo_names,
        index=0,
        key="combo_preset",
        help="Select combo for auto Primary+Confirm. Or 'Custom' to pick manually."
    )
    preset = COMBO_PRESETS[selected_combo]
    is_custom = selected_combo == "Custom"
    if not is_custom and preset.get("desc"):
        st.info(f"**{selected_combo}**\n\n{preset['desc']}")
    if is_custom:
        # --- CUSTOM MODE: User picks individually ---
        st.caption("Toggle conditions. More conditions = stronger confluence.")
        sig_sma = st.checkbox("SMA Cross (Golden/Death)", value=True, key="sig_sma")
        sig_macd = st.checkbox("MACD Cross Signal", value=True, key="sig_macd")
        sig_pullback = st.checkbox("EMA Pullback (EMA21)", value=False, key="sig_pullback")
        sig_bb_sq = st.checkbox("BB Squeeze Breakout", value=False, key="sig_bb_sq")
        sig_rsi_div = st.checkbox("RSI Divergence", value=False, key="sig_rsi_div")
        sig_macd_hist = st.checkbox("MACD Histogram Reversal", value=False, key="sig_macd_hist")
        sig_stoch = st.checkbox("Stochastic Cross (<20/>80)", value=False, key="sig_stoch")
        sig_bb_bounce = st.checkbox("BB Bounce (mean reversion)", value=False, key="sig_bb_bounce")
        sig_engulf = st.checkbox("Engulfing Candle", value=False, key="sig_engulf")
        st.markdown("---")
        st.caption("**High Win-Rate**")
        sig_ribbon = st.checkbox("EMA Ribbon 8/21/55 (trend ~60%)", value=False, key="sig_ribbon")
        sig_inside = st.checkbox("Inside Bar Breakout (R:R 3:1)", value=False, key="sig_inside")
        sig_hammer = st.checkbox("Hammer/Shooting Star (~60%)", value=False, key="sig_hammer")
        sig_adx = st.checkbox("ADX + DI Cross (trend filter)", value=False, key="sig_adx")
        sig_sr = st.checkbox("S/R Breakout 20D (R:R 3:1)", value=False, key="sig_sr")
        sig_sr_atr = st.checkbox("S/R ± ATR (volatility breakout)", value=False, key="sig_sr_atr")
        st.markdown("---")
        sig_vol = st.checkbox("Volume Filter (RVOL)", value=True, key="sig_vol")
        signal_enabled = {
            "sma_cross": sig_sma, "macd_cross": sig_macd,
            "ema_pullback": sig_pullback, "bb_squeeze": sig_bb_sq,
            "rsi_div": sig_rsi_div, "macd_hist_rev": sig_macd_hist,
            "stoch_cross": sig_stoch, "bb_bounce": sig_bb_bounce,
            "engulfing": sig_engulf, "ema_ribbon": sig_ribbon,
            "inside_bar": sig_inside, "hammer_star": sig_hammer,
            "adx_di": sig_adx, "sr_breakout": sig_sr,
            "sr_atr": sig_sr_atr, "vol_filter": sig_vol,
        }
        active_combo = None
    else:
        # --- COMBO MODE: Auto-enable from preset, ignore checkboxes ---
        all_primary = set(preset.get("primary", []))
        all_confirm = set(preset.get("confirm", []))
        all_gate = set(preset.get("gate", []))
        all_combo_keys = all_primary | all_confirm | all_gate
        st.caption("**Primary** = entry trigger | **Confirm** = double-check | **Gate** = blocking filter")
        # Show summary (read-only)
        for k in all_primary:
            st.markdown(f"  **[P] {k}**")
        for k in all_confirm:
            st.markdown(f"  [C] {k}")
        for k in all_gate:
            st.markdown(f"  [G] {k}")
        sig_vol = st.checkbox("Volume Filter (RVOL)", value=True, key="sig_vol_combo")
        # Force-enable all combo conditions
        signal_enabled = {k: (k in all_combo_keys) for k in [
            "sma_cross", "macd_cross", "ema_pullback", "bb_squeeze",
            "rsi_div", "macd_hist_rev", "stoch_cross", "bb_bounce",
            "engulfing", "ema_ribbon", "inside_bar", "hammer_star",
            "adx_di", "sr_breakout", "sr_atr", "macd_filter", "vol_color_filter",
        ]}
        signal_enabled["vol_filter"] = sig_vol
        active_combo = selected_combo
    # Capital
    initial_capital = st.number_input(
        "Initial Capital (VND)", value=100_000_000, step=10_000_000, format="%d"
    )
    # Apply button
    applied = st.form_submit_button("Apply", type="primary", width="stretch")
# Clear cache on Apply to force fresh data reload
if applied:
    st.cache_data.clear()
    # Clear stored signals to force recomputation
    st.session_state.pop("_combined_df", None)
    st.session_state.pop("_signals_computed", None)
# Build all strategies
all_strategies = [
    MACrossStrategy(fast_period=fast_ma, slow_period=slow_ma),
    RSIStrategy(period=rsi_period, oversold=oversold, overbought=overbought),
    MACDStrategy(fast=macd_fast, slow=macd_slow, signal=macd_signal),
]
# Strategy colors for markers
STRATEGY_COLORS = {
    "MA_Cross": {"buy": "#00E676", "sell": "#FF5252", "shape_buy": "arrowUp", "shape_sell": "arrowDown"},
    "RSI": {"buy": "#40C4FF", "sell": "#FF6E40", "shape_buy": "circle", "shape_sell": "circle"},
    "MACD": {"buy": "#E040FB", "sell": "#FFD740", "shape_buy": "square", "shape_sell": "square"},
}
# --- Data Loading ---
@st.cache_data(ttl=300)

def load_data(sym: str, start: str, end: str, mtype: str, intv: str = "1D"):
    fetcher = DataFetcher()
    if mtype == "Futures":
        return fetcher.get_futures_ohlcv(sym, start, end, interval=intv)
    elif mtype == "Index":
        return fetcher.get_index_ohlcv(sym, start, end, interval=intv)
    else:
        return fetcher.get_historical_ohlcv(sym, start, end, interval=intv)
# --- Main Content ---
try:
    df = load_data(symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), market_type, interval)
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()
if df is None or df.empty:
    st.warning("No data for this symbol.")
    st.stop()
# --- Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Price Chart", "Compare Strategies", "Backtest", "Multi-Symbol Scan", "Futures", "Raw Data", "Indicators"
])

# ============================

# TAB 1: Price Chart (TradingView style)

# ============================

with tab1:
    st.subheader(f"{symbol}  |  {interval}  |  {date_range}")
    # Indicator toggles
    selected_indicators = st.multiselect(
        "Technical Indicators",
        ["EMA 200", "MA Cross", "Bollinger Bands", "RSI", "MACD"],
        default=["EMA 200", "MA Cross", "RSI", "MACD"],
        key="tab1_indicators",
    )
    # Generate combined RSI + SMA Cross signals
    show_signals = st.checkbox("Show entry/exit signals", value=True, key="show_signals")
    signals_list = None
    combined_df = st.session_state.get("_combined_df")
    if show_signals:
        if combined_df is None or "_signals_computed" not in st.session_state:
            combined_df = generate_combined_signals(
                df, fast_ma=fast_ma, slow_ma=slow_ma,
                rsi_period=rsi_period, oversold=oversold, overbought=overbought,
                macd_fast=macd_fast, macd_slow=macd_slow, macd_signal=macd_signal,
                vol_mult=vol_mult, enabled=signal_enabled, combo_mode=active_combo,
            )
            st.session_state["_combined_df"] = combined_df
            st.session_state["_signals_computed"] = True

            # --- Telegram alert on new signal (last bar only) ---
            _last = combined_df.iloc[-1]
            _last_sig = int(_last.get("signal", 0))
            _last_sig_key = f"{symbol}_{_last.name}_{_last_sig}"
            if _last_sig != 0 and st.session_state.get("_last_alert_key") != _last_sig_key:
                _notifier = TelegramNotifier()
                if _notifier.is_configured():
                    _direction = "BUY" if _last_sig == 1 else "SELL"
                    _price = float(_last["close"])
                    _atr = float(_last.get("atr", 0))
                    _confidence = int(_last.get("signal_confidence", 0))
                    # Collect which conditions fired
                    _prefix = "_b_" if _last_sig == 1 else "_s_"
                    _fired = [COND_LABELS.get(k, k) for k in ALL_COND_KEYS
                              if _last.get(f"{_prefix}{k}", 0) == 1]
                    # SL/TP based on ATR
                    if _last_sig == 1:
                        _sl = _price - 1.5 * _atr
                        _tp = _price + 3.0 * _atr
                    else:
                        _sl = _price + 1.5 * _atr
                        _tp = _price - 3.0 * _atr
                    # Extra context
                    _extra = {
                        "RSI": f"{_last.get('rsi', 0):.1f}",
                        "EMA Slope": f"{_last.get('ema_slope', 0):.3f}",
                        "ADX": f"{_last.get('adx', 0):.1f}",
                        "ATR": f"{_atr:.2f}",
                    }
                    _notifier.send_signal_alert(
                        symbol=symbol, signal=_direction, price=_price,
                        conditions_fired=_fired, confidence=_confidence,
                        combo_name=active_combo or "Custom",
                        sl=_sl, tp=_tp, extra=_extra,
                    )
                    st.session_state["_last_alert_key"] = _last_sig_key
        signals_list = [
            ("Signal", {"buy": "#00E676", "sell": "#FF5252", "shape_buy": "arrowUp", "shape_sell": "arrowDown"}, combined_df),
        ]
        # Dynamic legend
        _cond_labels = COND_LABELS
        _active = [v for k, v in _cond_labels.items() if signal_enabled.get(k)]
        _n = sum(1 for k, v in signal_enabled.items() if v and k != "vol_filter")
        _vol_txt = f" + Vol>{vol_mult}x" if signal_enabled.get("vol_filter") else ""
        if active_combo:
            # Combo mode: show primary/confirm breakdown
            _p_labels = [_cond_labels.get(k, k) for k in COMBO_PRESETS[active_combo].get("primary", []) if signal_enabled.get(k)]
            _c_labels = [_cond_labels.get(k, k) for k in COMBO_PRESETS[active_combo].get("confirm", []) if signal_enabled.get(k)]
            st.markdown(
                f"**{active_combo}** | "
                f"Primary: {', '.join(_p_labels)} | "
                f"Confirm: {', '.join(_c_labels)}{_vol_txt}"
            )
            # Confidence stats
            n_signals = (combined_df["signal"] != 0).sum()
            if n_signals > 0:
                avg_conf = combined_df.loc[combined_df["signal"] != 0, "signal_confidence"].mean()
                high_conf = (combined_df.loc[combined_df["signal"] != 0, "signal_confidence"] >= 3).sum()
                conf_stars = "*" * min(3, int(round(avg_conf)))
                st.markdown(f"**{n_signals}** signals | Avg confidence: {conf_stars} ({avg_conf:.1f}/3) | *** HIGH: **{high_conf}**")
        else:
            _min_sc = max(1, _n // 3)
            st.markdown(
                f"**Active conditions:** {', '.join(_active) if _active else 'None'} "
                f"| Min score: **{_min_sc}**{_vol_txt}"
            )
    # Render TradingView lightweight-charts
    render_tv_chart(
        df, title=f"{symbol} | {interval}",
        indicators=selected_indicators,
        signals_list=signals_list,
        chart_key="main_chart",
    )
    # Summary stats
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Current Price", f"{df['close'].iloc[-1]:,.2f}")
    change = df["close"].iloc[-1] - df["close"].iloc[-2] if len(df) > 1 else 0
    change_pct = change / df["close"].iloc[-2] * 100 if len(df) > 1 else 0
    col2.metric("Change", f"{change:+,.2f}", f"{change_pct:+.2f}%")
    col3.metric("Highest", f"{df['high'].max():,.2f}")
    col4.metric("Lowest", f"{df['low'].min():,.2f}")
    col5.metric("TB Volume", f"{df['volume'].mean():,.0f}")
    # Signal detail table for all strategies
    if show_signals and signals_list:
        for label, colors, sig_df in signals_list:
            analysis = analyze_signal_performance(sig_df, atr_sl_mult=1.5, atr_tp_mult=3.0, max_hold=30)
            if analysis is None:
                continue
            adf = analysis["rows"]
            st.markdown("---")
            st.subheader("Signal Analysis")
            # Summary cards
            st.markdown("### Win/Loss Stats (backtest on current data)")
            sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
            sc1.metric("Total Signals", analysis["total"])
            sc2.metric("Win", analysis["wins"], f"{analysis['win_rate']:.1f}%")
            sc3.metric("Loss", analysis["losses"])
            sc4.metric("TB Win", f"+{analysis['avg_win']:.2f}%")
            sc5.metric("TB Loss", f"-{analysis['avg_loss']:.2f}%")
            sc6.metric("Profit Factor", f"{analysis['profit_factor']:.2f}")
            # Expectancy
            exp_color = "green" if analysis["expectancy"] > 0 else "red"
            st.markdown(
                f"**Expectancy (expectancy/trade):** "
                f"<span style='color:{exp_color};font-size:1.2em;font-weight:bold'>{analysis['expectancy']:+.2f}%</span> "
                f"&nbsp;|&nbsp; SL = 1.5×ATR &nbsp;|&nbsp; TP = 3×ATR &nbsp;|&nbsp; "
                f"R:R = 1:2.0 &nbsp;|&nbsp; Max hold = 30 bars",
                unsafe_allow_html=True,
            )
            if analysis["expectancy"] > 0:
                st.success(f"POSITIVE expectancy: avg +{analysis['expectancy']:.2f}% per trade")
            elif analysis["wins"] + analysis["losses"] > 0:
                st.error(f"NEGATIVE expectancy: avg -{abs(analysis['expectancy']):.2f}% per trade")
            # Detail table
            st.markdown("### Trade Details")
            display_cols = ["Time", "Type", "Entry", "Stop Loss", "Take Profit",
                            "R:R", "Confidence", "Result", "P&L %"]
            st.dataframe(
                adf[display_cols].style.map(
                    lambda v: "color: #26a69a" if "WIN" in str(v) else ("color: #ef5350" if "LOSS" in str(v) else ""),
                    subset=["Result"]
                ).map(
                    lambda v: "color: #26a69a" if str(v).startswith("+") else
                              ("color: #ef5350" if str(v).startswith("-") else ""),
                    subset=["P&L %"]
                ),
                width="stretch", hide_index=True,
            )
            # --- Exit strategy explanation ---
            with st.expander("Exit Strategy"):
                st.markdown("""
**Based on Investopedia - Risk Management for Active Traders:**

**1. Stop Loss = 1.5 x ATR(14)**
- ATR = Average True Range 14 bars (avg volatility)
- SL placed 1.5x ATR from entry to avoid noise whipsaws

**2. Take Profit = 3.0 x ATR(14)**
- R:R = 1:2 so each win covers 2 losses
- Win rate 40% is still profitable if R:R >= 1:2

**3. Chandelier Exit (Trailing Stop)**
- Trailing stop = Highest high since entry - 2xATR
- As price moves favorably, SL trails to protect profits

**4. Max hold: 30 bars**
- No TP/SL hit in 30 bars -> exit at current price
- Avoids holding too long in sideways markets

**5. One-Percent Rule**
- Never risk more than 1% of total capital per trade

**When to EXIT a BUY:**
- Price hits Stop Loss -> cut loss immediately
- Price hits Take Profit -> take profit
- Opposite SELL signal -> exit early
- 30 bars elapsed -> exit at market price
""")

# ============================

# TAB 2: Strategy Comparison (Compare Strategies)

# ============================

with tab2:
    st.subheader(f"Compare Strategies — {symbol} | {interval}")
    st.caption("Run all combo presets on same data, analyze and compare performance.")
    # Gather presets (skip Custom)
    _compare_presets = {k: v for k, v in COMBO_PRESETS.items() if k != "Custom"}
    _compare_results = {}
    # Run all 4 strategies
    for preset_name, preset_cfg in _compare_presets.items():
        _preset_enabled = {}
        for cond in preset_cfg.get("primary", []) + preset_cfg.get("confirm", []) + preset_cfg.get("gate", []):
            _preset_enabled[cond] = True
        _sig_df = generate_combined_signals(
            df, fast_ma=fast_ma, slow_ma=slow_ma,
            rsi_period=rsi_period, oversold=oversold, overbought=overbought,
            macd_fast=macd_fast, macd_slow=macd_slow, macd_signal=macd_signal,
            vol_mult=vol_mult, enabled=_preset_enabled, combo_mode=preset_name,
        )
        _result = analyze_signal_performance(_sig_df, atr_sl_mult=1.5, atr_tp_mult=3.0, max_hold=30)
        _compare_results[preset_name] = {"sig_df": _sig_df, "analysis": _result, "cfg": preset_cfg}
    # ---- Individual strategy analysis ----
    st.markdown("---")
    st.markdown("## Analysis per strategy")
    for preset_name, res in _compare_results.items():
        analysis = res["analysis"]
        cfg = res["cfg"]
        short_name = preset_name.split(":")[0].strip()
        desc = cfg.get("desc", "")
        primary_str = ", ".join(cfg.get("primary", []))
        confirm_str = ", ".join(cfg.get("confirm", []))
        gate_str = ", ".join(cfg.get("gate", []))
        with st.expander(f"**{preset_name}**", expanded=False):
            st.markdown(f"*{desc}*")
            _info = f"**Primary:** `{primary_str}` &nbsp;|&nbsp; **Confirm:** `{confirm_str}`"
            if gate_str:
                _info += f" &nbsp;|&nbsp; **Gate:** `{gate_str}`"
            st.markdown(_info)
            if analysis is None:
                st.warning("No signals in this time period.")
                continue
            adf = analysis["rows"]
            # Summary metrics
            mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
            mc1.metric("Total Trades", analysis["total"])
            mc2.metric("Win", analysis["wins"], f"{analysis['win_rate']:.1f}%")
            mc3.metric("Loss ", analysis["losses"])
            mc4.metric("TB Win", f"+{analysis['avg_win']:.2f}%")
            mc5.metric("TB Loss", f"-{analysis['avg_loss']:.2f}%")
            mc6.metric("Profit Factor", f"{analysis['profit_factor']:.2f}")
            exp_color = "green" if analysis["expectancy"] > 0 else "red"
            st.markdown(
                f"**Expectancy:** <span style='color:{exp_color};font-size:1.1em;font-weight:bold'>"
                f"{analysis['expectancy']:+.2f}%</span> &nbsp;|&nbsp; "
                f"**Total P&L:** <span style='color:{exp_color};font-weight:bold'>{analysis['total_pnl']:+.2f}%</span>",
                unsafe_allow_html=True,
            )
            # Detail table
            display_cols = ["Time", "Type", "Entry", "Stop Loss", "Take Profit",
                            "R:R", "Confidence", "Result", "P&L %"]
            st.dataframe(
                adf[display_cols].style.map(
                    lambda v: "color: #26a69a" if "WIN" in str(v) else ("color: #ef5350" if "LOSS" in str(v) else ""),
                    subset=["Result"]
                ).map(
                    lambda v: "color: #26a69a" if str(v).startswith("+") else ("color: #ef5350" if str(v).startswith("-") else ""),
                    subset=["P&L %"]
                ),
                width="stretch", hide_index=True,
            )
    # ---- Comparison Table ----
    st.markdown("---")
    st.markdown("## Comparison Table")
    compare_rows = []
    for preset_name, res in _compare_results.items():
        a = res["analysis"]
        short_name = preset_name.split(":")[0].strip()
        if a is None:
            compare_rows.append({
                "Strategy": preset_name,
                "Total Trades": 0, "Win": 0, "Loss": 0,
                "Win Rate": "—", "TB Win": "—", "TB Loss": "—",
                "Profit Factor": "—", "Expectancy": "—", "Total P&L": "—",
                "Rating": "- No signals",
                "_exp": -999, "_wr": 0,
            })
        else:
            # Rating logic
            rating = ""
            if a["expectancy"] > 0 and a["win_rate"] >= 50 and a["profit_factor"] >= 1.5:
                rating = "Excellent"
            elif a["expectancy"] > 0 and a["profit_factor"] >= 1.0:
                rating = "Good"
            elif a["expectancy"] > 0:
                rating = "Acceptable"
            elif a["total"] < 3:
                rating = "- Low data"
            else:
                rating = "Poor"
            compare_rows.append({
                "Strategy": preset_name,
                "Total Trades": a["total"],
                "Win": a["wins"],
                "Loss": a["losses"],
                "Win Rate": f"{a['win_rate']:.1f}%",
                "TB Win": f"+{a['avg_win']:.2f}%",
                "TB Loss": f"-{a['avg_loss']:.2f}%",
                "Profit Factor": f"{a['profit_factor']:.2f}",
                "Expectancy": f"{a['expectancy']:+.2f}%",
                "Total P&L": f"{a['total_pnl']:+.2f}%",
                "Rating": rating,
                "_exp": a["expectancy"], "_wr": a["win_rate"],
            })
    cdf = pd.DataFrame(compare_rows)
    # Sort by expectancy descending
    cdf = cdf.sort_values("_exp", ascending=False)
    display_compare = ["Strategy", "Total Trades", "Win", "Loss", "Win Rate",
                       "TB Win", "TB Loss", "Profit Factor", "Expectancy", "Total P&L", "Rating"]
    st.dataframe(
        cdf[display_compare].style.map(
            lambda v: "color: #26a69a; font-weight: bold" if "Excellent" in str(v) or "Good" in str(v)
                      else ("color: #ef5350; font-weight: bold" if "Poor" in str(v) else ""),
            subset=["Rating"]
        ).map(
            lambda v: "color: #26a69a" if str(v).startswith("+") else ("color: #ef5350" if str(v).startswith("-") else ""),
            subset=["Expectancy", "Total P&L"]
        ),
        width="stretch", hide_index=True, height=220,
    )
    # ---- Best strategy recommendation ----
    best = cdf.iloc[0]
    if best["_exp"] > -999:
        st.markdown("---")
        if best["_exp"] > 0:
            st.success(
                f"### Best Strategy: **{best['Strategy']}**\n"
                f"- Win Rate: **{best['Win Rate']}** | Expectancy: **{best['Expectancy']}** | "
                f"Profit Factor: **{best['Profit Factor']}** | Total P&L: **{best['Total P&L']}**"
            )
        else:
            st.warning(
                f"### No strategy has positive expectancy on {symbol}\n"
                f"Least bad strategy: **{best['Strategy']}** (Expectancy: {best['Expectancy']})"
            )
    # ---- Win Rate comparison bar chart ----
    st.markdown("### Visual Comparison")
    chart_data = cdf[cdf["_exp"] > -999].copy()
    if not chart_data.empty:
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            fig_wr = go.Figure()
            colors_wr = [TV_UP if float(str(r).replace("%","")) >= 50 else TV_DOWN
                         for r in chart_data["Win Rate"]]
            fig_wr.add_trace(go.Bar(
                x=[n.split(":")[0].strip() for n in chart_data["Strategy"]],
                y=[float(str(r).replace("%","")) for r in chart_data["Win Rate"]],
                marker_color=colors_wr,
                text=chart_data["Win Rate"],
                textposition="outside",
            ))
            fig_wr.update_layout(
                title="Win Rate (%)", yaxis_title="%",
                **TV_LAYOUT, height=350,
            )
            fig_wr.update_xaxes(**TV_XAXIS)
            fig_wr.update_yaxes(**TV_YAXIS, range=[0, 100])
            st.plotly_chart(fig_wr, width="stretch")
        with chart_col2:
            fig_exp = go.Figure()
            colors_exp = [TV_UP if e >= 0 else TV_DOWN for e in chart_data["_exp"]]
            fig_exp.add_trace(go.Bar(
                x=[n.split(":")[0].strip() for n in chart_data["Strategy"]],
                y=chart_data["_exp"].tolist(),
                marker_color=colors_exp,
                text=chart_data["Expectancy"],
                textposition="outside",
            ))
            fig_exp.update_layout(
                title="Expectancy (%/trade)", yaxis_title="%",
                **TV_LAYOUT, height=350,
            )
            fig_exp.update_xaxes(**TV_XAXIS)
            fig_exp.update_yaxes(**TV_YAXIS)
            st.plotly_chart(fig_exp, width="stretch")
    st.caption("SL = 1.5×ATR | TP = 3×ATR | R:R = 1:2 | Max hold = 30 bars | Data: " + date_range)

# ============================

# TAB 3: Backtest

# ============================

with tab3:
    st.subheader(f"Backtest: {symbol} — All Strategies")
    backtester = Backtester(initial_capital=initial_capital)
    results = []
    for strat in all_strategies:
        result = backtester.run(df, strat)
        results.append(result)
    # Equity curve chart
    fig_equity = go.Figure()
    for i, result in enumerate(results):
        fig_equity.add_trace(
            go.Scatter(
                y=result.equity_curve.values,
                name=result.stats["strategy"],
                mode="lines",
            )
        )
    fig_equity.add_hline(y=initial_capital, line_dash="dash", line_color="gray", annotation_text="Initial Capital")
    fig_equity.update_layout(
        title="Equity Curve",
        yaxis_title="Portfolio Value (VND)",
        xaxis_title="Trading Days",
        height=400,
        **TV_LAYOUT,
    )
    fig_equity.update_xaxes(TV_XAXIS)
    fig_equity.update_yaxes(TV_YAXIS)
    st.plotly_chart(fig_equity, width="stretch")
    # Stats table
    st.subheader("Detailed Results")
    stats_data = []
    for result in results:
        s = result.stats
        stats_data.append({
            "Strategy": s["strategy"],
            "Trades": s["total_trades"],
            "Win Rate (%)": f"{s['win_rate']:.1f}",
            "Total Return (%)": f"{s['total_return']:.2f}",
            "Max Drawdown (%)": f"{s['max_drawdown']:.2f}",
            "Sharpe Ratio": f"{s['sharpe_ratio']:.2f}",
            "Profit Factor": f"{s['profit_factor']:.2f}",
            "Avg Win (%)": f"{s['avg_win']:.2f}",
            "Avg Loss (%)": f"{s['avg_loss']:.2f}",
        })
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, width="stretch", hide_index=True)
    # Trade history
    if results:
        best_result = max(results, key=lambda r: r.stats["total_return"])
        if not best_result.trades.empty:
            st.subheader(f"Trade History - {best_result.stats['strategy']}")
            st.dataframe(best_result.trades, width="stretch", hide_index=True)

# ============================

# TAB 4: Multi-stock Scan

# ============================

with tab4:
    st.subheader("Multi-Symbol Signal Scan")
    default_symbols = "HPG, FPT, VNM, VCB, MWG, TCB, ACB, VIC, VHM, MSN"
    symbols_input = st.text_input("Enter symbols (comma separated)", value=default_symbols)
    symbols_list = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    scan_strategy_name = st.selectbox(
        "Strategy scan",
        ["MA Crossover 10/20", "RSI 14", "MACD 12/26/9"],
        key="scan_strategy",
    )
    if st.button("Start Scan", type="primary"):
        if scan_strategy_name == "MA Crossover 10/20":
            scan_strategy = MACrossStrategy(10, 20)
        elif scan_strategy_name == "RSI 14":
            scan_strategy = RSIStrategy(14)
        else:
            scan_strategy = MACDStrategy()
        fetcher = DataFetcher()
        scan_end = datetime.now().strftime("%Y-%m-%d")
        scan_start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        scan_results = []
        progress = st.progress(0)
        for i, sym in enumerate(symbols_list):
            try:
                sym_df = fetcher.get_historical_ohlcv(sym, scan_start, scan_end)
                df_sig = scan_strategy.generate_signals(sym_df)
                last_signal = df_sig["signal"].iloc[-1]
                last_price = df_sig["close"].iloc[-1]
                signal_text = "BUY" if last_signal == 1 else "SELL" if last_signal == -1 else "- HOLD"
                scan_results.append({
                    "Symbol": sym,
                    "Price": f"{last_price:,.2f}",
                    "Signal": signal_text,
                    "Signal_val": last_signal,
                })
            except Exception as e:
                scan_results.append({
                    "Symbol": sym,
                    "Price": "N/A",
                    "Signal": f"! Error: {e}",
                    "Signal_val": 0,
                })
            progress.progress((i + 1) / len(symbols_list))
        progress.empty()
        if scan_results:
            scan_df = pd.DataFrame(scan_results)
            # Highlight BUY/SELL
            buy_df = scan_df[scan_df["Signal"] == 1]
            sell_df = scan_df[scan_df["Signal"] == -1]
            if not buy_df.empty:
                st.success(f"BUY Signal: {', '.join(buy_df['Symbol'].tolist())}")
            if not sell_df.empty:
                st.error(f"SELL Signal: {', '.join(sell_df['Symbol'].tolist())}")
            st.dataframe(
                scan_df[["Symbol", "Price", "Signal"]],
                width="stretch",
                hide_index=True,
            )

# ============================

# TAB 5: Futures (Futures)

# ============================

with tab5:
    st.subheader("Futures Trading")
    futures_col1, futures_col2 = st.columns([1, 3])
    with futures_col1:
        futures_symbol = st.selectbox(
            "Contract",
            ["VN30F1M", "VN30F2M", "VN30F1Q", "VN30F2Q"],
            key="futures_select",
        )
        futures_days = st.selectbox("Period", [30, 60, 90, 180, 365], index=2, key="futures_days")
        # Futures info
        st.markdown("---")
        st.markdown("""
        **Futures Info:**
        - **VN30F1M**: Nearest month contract
        - **VN30F2M**: Next month contract
        - **VN30F1Q**: Nearest quarter contract
        - **VN30F2Q**: Next quarter contract
        - Session: 8:45 - 14:45
        - Tick size: 0.1 pts
        - Multiplier: 100,000 VND

        """)
    with futures_col2:
        try:
            fetcher_f = DataFetcher()
            f_end = datetime.now().strftime("%Y-%m-%d")
            f_start = (datetime.now() - timedelta(days=futures_days)).strftime("%Y-%m-%d")
            df_futures = fetcher_f.get_futures_ohlcv(futures_symbol, f_start, f_end)
            if df_futures is not None and not df_futures.empty:
                futures_indicators = st.multiselect(
                    "Technical Indicators",
                    ["EMA 200", "MA Cross", "Bollinger Bands", "RSI", "MACD"],
                    default=["MA Cross", "RSI", "MACD"],
                    key="futures_indicators",
                )
                # Generate combined signals for futures
                f_combined = generate_combined_signals(
                    df_futures, fast_ma=fast_ma, slow_ma=slow_ma,
                    rsi_period=rsi_period, oversold=oversold, overbought=overbought,
                    macd_fast=macd_fast, macd_slow=macd_slow, macd_signal=macd_signal,
                    vol_mult=vol_mult, enabled=signal_enabled, combo_mode=active_combo,
                )
                f_signals_list = [
                    ("Signal", {"buy": "#00E676", "sell": "#FF5252", "shape_buy": "arrowUp", "shape_sell": "arrowDown"}, f_combined),
                ]
                render_tv_chart(
                    df_futures, title=f"{futures_symbol} | Futures",
                    indicators=futures_indicators,
                    signals_list=f_signals_list,
                    chart_key="futures_chart",
                )
                # Stats
                fcol1, fcol2, fcol3, fcol4 = st.columns(4)
                fcol1.metric("Close", f"{df_futures['close'].iloc[-1]:,.1f}")
                f_change = df_futures["close"].iloc[-1] - df_futures["close"].iloc[-2] if len(df_futures) > 1 else 0
                fcol2.metric("Change", f"{f_change:+,.1f}")
                fcol3.metric("Highest (period)", f"{df_futures['high'].max():,.1f}")
                fcol4.metric("Lowest (period)", f"{df_futures['low'].min():,.1f}")
                # Backtest on futures
                st.markdown("---")
                st.subheader(f"Backtest {futures_symbol}")
                f_strategy = MACrossStrategy(fast_period=5, slow_period=15)
                f_backtester = Backtester(initial_capital=50_000_000)
                f_result = f_backtester.run(df_futures, f_strategy)
                st.text(f_result.summary())
            else:
                st.warning("No futures data.")
        except Exception as e:
            st.error(f"Error fetching data futures: {e}")

# ============================

# TAB 6: Raw Data

# ============================

with tab6:
    st.subheader(f"Raw Data: {symbol}")
    st.dataframe(df, width="stretch", hide_index=True)
    # Download CSV
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{interval}.csv",
        mime="text/csv",
    )
    # Basic stats
    st.subheader("Basic Statistics")
    st.dataframe(df[["open", "high", "low", "close", "volume"]].describe(), width="stretch")

# ============================

# TAB 7: Indicators (debug/verification)

# ============================

with tab7:
    _ind_df = st.session_state.get("_combined_df")
    if _ind_df is not None and not _ind_df.empty:
        last = _ind_df.iloc[-1]
        st.caption(f"{symbol} | {interval} | Last bar")

        # --- S/R ---
        st.markdown("---")
        st.markdown("**Support / Resistance**")
        sr_data = {
            "Metric": ["Dynamic Resistance", "Dynamic Support", "S/R Strength", "Resistance (20)", "Support (20)", "Close"],
            "Value": [
                f"{last.get('dynamic_resistance', 0):,.2f}",
                f"{last.get('dynamic_support', 0):,.2f}",
                f"{last.get('sr_strength', 0):.0f}",
                f"{last.get('resistance_20', 0):,.2f}",
                f"{last.get('support_20', 0):,.2f}",
                f"{last['close']:,.2f}",
            ]
        }
        st.dataframe(sr_data, hide_index=True, use_container_width=True)

        # --- EMA & Trend ---
        st.markdown("---")
        st.markdown("**EMA & Trend**")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.caption("EMA5"); c1.code(f"{last.get('ema5', 0):,.2f}")
        c2.caption("EMA12"); c2.code(f"{last.get('ema12', 0):,.2f}")
        c3.caption("EMA21"); c3.code(f"{last.get('ema21', 0):,.2f}")
        c4.caption("EMA55"); c4.code(f"{last.get('ema55', 0):,.2f}")
        c5.caption("Slope"); c5.code(f"{last.get('ema_slope', 0):,.3f}")
        _trend = "BULL" if last.get('trend_bull') else ("BEAR" if last.get('trend_bear') else "--")
        c6.caption("Trend"); c6.code(_trend)

        # --- Momentum ---
        st.markdown("---")
        st.markdown("**Momentum (RSI / Stoch / MACD)**")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.caption("RSI"); c1.code(f"{last.get('rsi', 0):.1f}")
        c2.caption("%K"); c2.code(f"{last.get('stoch_k', 0):.1f}")
        c3.caption("%D"); c3.code(f"{last.get('stoch_d', 0):.1f}")
        c4.caption("MACD"); c4.code(f"{last.get('macd_line', 0):.4f}")
        c5.caption("Signal"); c5.code(f"{last.get('macd_sig', 0):.4f}")
        c6.caption("Hist"); c6.code(f"{last.get('macd_hist', 0):.4f}")

        # --- Volatility ---
        st.markdown("---")
        st.markdown("**Volatility (BB / ATR / Z-Score)**")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.caption("BB Upper"); c1.code(f"{last.get('bb_upper', 0):,.2f}")
        c2.caption("BB Lower"); c2.code(f"{last.get('bb_lower', 0):,.2f}")
        c3.caption("BB Width%"); c3.code(f"{last.get('bb_width_pct', 0)*100:.2f}%")
        c4.caption("Squeeze"); c4.code("Y" if last.get('bb_squeeze_flag') else "N")
        c5.caption("ATR(14)"); c5.code(f"{last.get('atr', 0):,.2f}")
        c6.caption("Z-Score"); c6.code(f"{last.get('zscore', 0):.2f}")

        # --- ADX & Volume ---
        st.markdown("---")
        st.markdown("**ADX & Volume**")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.caption("ADX"); c1.code(f"{last.get('adx', 0):.1f}")
        c2.caption("+DI"); c2.code(f"{last.get('plus_di', 0):.1f}")
        c3.caption("-DI"); c3.code(f"{last.get('minus_di', 0):.1f}")
        c4.caption("Volume"); c4.code(f"{last.get('volume', 0):,.0f}")
        c5.caption("Vol SMA"); c5.code(f"{last.get('vol_sma', 0):,.0f}")
        c6.caption("Vol OK"); c6.code("Y" if last.get('vol_ok') else "N")

        # --- Signal Summary ---
        st.markdown("---")
        st.markdown("**Signal**")
        c1, c2, c3, c4 = st.columns(4)
        _sig = last.get('signal', 0)
        _sig_txt = "**BUY**" if _sig == 1 else ("**SELL**" if _sig == -1 else "HOLD")
        c1.caption("Direction"); c1.markdown(_sig_txt)
        c2.caption("Confidence"); c2.code(f"{last.get('signal_confidence', 0)}")
        c3.caption("Buy Score"); c3.code(f"{last.get('buy_score', 0)}")
        c4.caption("Sell Score"); c4.code(f"{last.get('sell_score', 0)}")

        # --- Table: last 10 bars ---
        st.markdown("---")
        st.markdown("**Last 10 bars**")
        show_cols = ["time", "close", "rsi", "macd_hist", "ema_slope", "zscore",
                     "atr", "adx", "dynamic_resistance", "dynamic_support",
                     "signal", "signal_confidence"]
        available_cols = [c for c in show_cols if c in _ind_df.columns]
        st.dataframe(_ind_df[available_cols].tail(10), width="stretch", hide_index=True)
    else:
        st.info("No indicator data. Click Apply to generate signals.")
