"""Strategy 20: Renko Brick Reversal.

Converts close prices from the last `lookback` bars into Renko bricks using
brick_size = brick_size_mult * ATR at the current bar.

  BUY:  first up-brick after 2+ consecutive down-bricks
  SELL: first down-brick after 2+ consecutive up-bricks

Because ATR is recomputed at each bar, the brick size is adaptive.  The Renko
sequence is built fresh at every detect() call — no persistent state needed.
"""
import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


class RenkoStrategy(BaseStrategy):
    name = "renko"
    exit_type = "trend"

    def __init__(self, brick_size_mult=1.0, lookback=50, min_bricks=2):
        """
        Args:
            brick_size_mult: ATR multiplier for brick size
            lookback: number of recent close prices used to build the brick sequence
            min_bricks: consecutive opposite-direction bricks required before reversal
        """
        self.brick_size_mult = brick_size_mult
        self.lookback = lookback
        self.min_bricks = min_bricks

    def _build_renko(self, closes, brick_size):
        """Build Renko brick sequence from close prices.

        Returns list of +1 (up brick) and -1 (down brick).
        """
        bricks = []
        if len(closes) == 0 or brick_size <= 0:
            return bricks

        current_level = closes[0]
        for price in closes[1:]:
            # Up bricks
            while price >= current_level + brick_size:
                bricks.append(1)
                current_level += brick_size
            # Down bricks
            while price <= current_level - brick_size:
                bricks.append(-1)
                current_level -= brick_size

        return bricks

    def detect(self, df, idx):
        if idx < self.lookback:
            return 0

        atr = df['atr'].iloc[idx]
        if pd.isna(atr) or atr <= 0:
            return 0

        brick_size = self.brick_size_mult * atr
        closes = df['close'].iloc[idx - self.lookback: idx + 1].values
        bricks = self._build_renko(closes, brick_size)

        # Need at least (min_bricks opposing) + (1 current) bricks
        if len(bricks) < self.min_bricks + 1:
            return 0

        last_brick = bricks[-1]
        prior = bricks[:-1]

        # BUY: current brick is up (+1), count consecutive downs before it
        if last_brick == 1:
            consec_down = 0
            for b in reversed(prior):
                if b == -1:
                    consec_down += 1
                else:
                    break
            if consec_down >= self.min_bricks:
                # PM BUY direction shows PF=0.46 on 1m data — suppress
                if 'session' in df.columns and df['session'].iloc[idx] == 'PM':
                    return 0
                return 1

        # SELL: current brick is down (-1), count consecutive ups before it
        elif last_brick == -1:
            consec_up = 0
            for b in reversed(prior):
                if b == 1:
                    consec_up += 1
                else:
                    break
            if consec_up >= self.min_bricks:
                return -1

        return 0
