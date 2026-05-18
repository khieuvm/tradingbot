"""
Backtest module - Kiểm tra chiến lược trên dữ liệu lịch sử
"""
import pandas as pd
import numpy as np
from src.strategy import BaseStrategy


class BacktestResult:
    """Kết quả backtest"""

    def __init__(self, trades: pd.DataFrame, equity_curve: pd.Series, stats: dict):
        self.trades = trades
        self.equity_curve = equity_curve
        self.stats = stats

    def summary(self) -> str:
        lines = [
            f"=== BACKTEST RESULT ===",
            f"Strategy: {self.stats['strategy']}",
            f"Period: {self.stats['start_date']} -> {self.stats['end_date']}",
            f"---",
            f"Total trades: {self.stats['total_trades']}",
            f"Win rate: {self.stats['win_rate']:.1f}%",
            f"Total return: {self.stats['total_return']:.2f}%",
            f"Max drawdown: {self.stats['max_drawdown']:.2f}%",
            f"Sharpe ratio: {self.stats['sharpe_ratio']:.2f}",
            f"Profit factor: {self.stats['profit_factor']:.2f}",
            f"Avg win: {self.stats['avg_win']:.2f}%",
            f"Avg loss: {self.stats['avg_loss']:.2f}%",
        ]
        return "\n".join(lines)


class Backtester:
    """Engine backtest chiến lược"""

    def __init__(self, initial_capital: float = 100_000_000, commission: float = 0.0):
        """
        Args:
            initial_capital: Vốn ban đầu (VND), mặc định 100 triệu
            commission: Phí giao dịch (%), DNSE miễn phí nên mặc định 0
        """
        self.initial_capital = initial_capital
        self.commission = commission

    def run(self, df: pd.DataFrame, strategy: BaseStrategy) -> BacktestResult:
        """Chạy backtest

        Args:
            df: DataFrame với dữ liệu OHLCV (columns: open, high, low, close, volume)
            strategy: Strategy object đã implement generate_signals()
        """
        # Generate signals
        df_signals = strategy.generate_signals(df)

        # Simulate trades
        capital = self.initial_capital
        position = 0  # Số lượng cổ phiếu đang nắm
        trades = []
        equity = [capital]
        entry_price = 0.0

        for i in range(len(df_signals)):
            row = df_signals.iloc[i]
            signal = row["signal"]
            price = row["close"]

            if signal == 1 and position == 0:
                # BUY
                qty = int(capital * 0.95 / price / 100) * 100  # Mua tối đa 95% vốn, làm tròn lô 100
                if qty > 0:
                    cost = qty * price * (1 + self.commission)
                    capital -= cost
                    position = qty
                    entry_price = price
                    trades.append({
                        "date": row.name if hasattr(row, 'name') else i,
                        "type": "BUY",
                        "price": price,
                        "qty": qty,
                        "value": cost,
                    })

            elif signal == -1 and position > 0:
                # SELL
                revenue = position * price * (1 - self.commission)
                pnl_pct = (price - entry_price) / entry_price * 100
                capital += revenue
                trades.append({
                    "date": row.name if hasattr(row, 'name') else i,
                    "type": "SELL",
                    "price": price,
                    "qty": position,
                    "value": revenue,
                    "pnl_pct": pnl_pct,
                })
                position = 0

            # Tính equity = cash + market value
            equity.append(capital + position * price)

        # Tính toán thống kê
        equity_series = pd.Series(equity[1:], index=df_signals.index)
        trades_df = pd.DataFrame(trades)
        stats = self._calc_stats(trades_df, equity_series, strategy.name, df_signals)

        return BacktestResult(trades=trades_df, equity_curve=equity_series, stats=stats)

    def _calc_stats(self, trades_df: pd.DataFrame, equity: pd.Series, strategy_name: str, df: pd.DataFrame) -> dict:
        """Tính toán các chỉ số thống kê"""
        sell_trades = trades_df[trades_df["type"] == "SELL"] if not trades_df.empty else pd.DataFrame()

        total_trades = len(sell_trades)
        wins = sell_trades[sell_trades["pnl_pct"] > 0] if not sell_trades.empty else pd.DataFrame()
        losses = sell_trades[sell_trades["pnl_pct"] <= 0] if not sell_trades.empty else pd.DataFrame()

        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
        total_return = (equity.iloc[-1] - self.initial_capital) / self.initial_capital * 100

        # Max drawdown
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()

        # Sharpe ratio (giả định risk-free = 5%/năm)
        daily_returns = equity.pct_change().dropna()
        sharpe = (daily_returns.mean() * 252 - 0.05) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0

        # Profit factor
        gross_profit = wins["pnl_pct"].sum() if not wins.empty else 0
        gross_loss = abs(losses["pnl_pct"].sum()) if not losses.empty else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_win = wins["pnl_pct"].mean() if not wins.empty else 0
        avg_loss = losses["pnl_pct"].mean() if not losses.empty else 0

        return {
            "strategy": strategy_name,
            "start_date": str(df.index[0]) if hasattr(df.index[0], 'strftime') else str(df.index[0]),
            "end_date": str(df.index[-1]) if hasattr(df.index[-1], 'strftime') else str(df.index[-1]),
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
        }
