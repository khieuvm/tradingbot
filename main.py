"""
Trading Bot - Main Entry Point
================================
Sử dụng:
    python main.py fetch HPG              # Lấy dữ liệu HPG
    python main.py backtest HPG           # Backtest strategy trên HPG
    python main.py scan HPG FPT VNM       # Scan tín hiệu realtime
    python main.py account                # Xem thông tin tài khoản
"""
import sys
from datetime import datetime, timedelta

from config import Config
from src.data_fetcher import DataFetcher
from src.strategy import MACrossStrategy, RSIStrategy, MACDStrategy
from src.backtest import Backtester
from src.bot import TradingBot
from src.notifier import TelegramNotifier


def cmd_fetch(symbols: list[str]):
    """Lấy và hiển thị dữ liệu"""
    fetcher = DataFetcher()
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"  {symbol} | {start} -> {end}")
        print(f"{'='*50}")
        df = fetcher.get_historical_ohlcv(symbol, start, end)
        print(df.tail(10).to_string())
        print(f"\nTotal rows: {len(df)}")


def cmd_backtest(symbols: list[str]):
    """Backtest strategies"""
    fetcher = DataFetcher()
    backtester = Backtester(initial_capital=100_000_000)

    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    strategies = [
        MACrossStrategy(fast_period=10, slow_period=20),
        RSIStrategy(period=14, oversold=30, overbought=70),
        MACDStrategy(fast=12, slow=26, signal=9),
    ]

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"  BACKTEST: {symbol} | {start} -> {end}")
        print(f"{'='*60}")

        df = fetcher.get_historical_ohlcv(symbol, start, end)

        for strategy in strategies:
            result = backtester.run(df, strategy)
            print(f"\n{result.summary()}")
            print()


def cmd_scan(symbols: list[str]):
    """Scan tín hiệu trading"""
    fetcher = DataFetcher()
    strategy = MACrossStrategy(fast_period=10, slow_period=20)
    bot = TradingBot(strategy=strategy)

    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    print(f"\n🔍 Scanning {len(symbols)} symbols with {strategy.name}...")
    print(f"{'='*50}")

    for symbol in symbols:
        try:
            df = fetcher.get_historical_ohlcv(symbol, start, end)
            signal = bot.check_signal(df, symbol)
            status = "🟢 BUY" if signal == 1 else "🔴 SELL" if signal == -1 else "⚪ HOLD"
            price = df["close"].iloc[-1]
            print(f"  {symbol:6s} | {price:>12,.0f} | {status}")
        except Exception as e:
            print(f"  {symbol:6s} | ERROR: {e}")


def cmd_account():
    """Hiển thị thông tin tài khoản"""
    Config.validate()
    bot = TradingBot(strategy=MACrossStrategy())

    print("\n📊 Account Information")
    print(f"{'='*50}")
    try:
        info = bot.get_account_info()
        print(info)
    except Exception as e:
        print(f"Error: {e}")


def cmd_price_board(symbols: list[str]):
    """Hiển thị bảng giá"""
    fetcher = DataFetcher()
    print(f"\n📈 Price Board: {', '.join(symbols)}")
    print(f"{'='*50}")
    df = fetcher.get_price_board(symbols)
    print(df.to_string())


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1].lower()
    args = sys.argv[2:]

    if command == "fetch":
        if not args:
            args = ["HPG"]
        cmd_fetch(args)

    elif command == "backtest":
        if not args:
            args = ["HPG"]
        cmd_backtest(args)

    elif command == "scan":
        if not args:
            args = ["HPG", "FPT", "VNM", "VCB", "MWG", "TCB", "ACB", "VIC", "VHM", "MSN"]
        cmd_scan(args)

    elif command == "account":
        cmd_account()

    elif command == "board":
        if not args:
            args = ["HPG", "FPT", "VNM", "VCB"]
        cmd_price_board(args)

    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
