"""
HermesQuant — Liquidity Sweep Engine
Based on LTDA research: 58% WR, PF 1.97, +28.2%/month on BTC 1H
"""
import numpy as np
import pandas as pd


class LiquidityDetector:
    """
    Phase 1: Find liquidity levels (swing highs/lows, clusters, EQH/EQL, PDH/PDL)
    """
    
    def __init__(self, cluster_pct=0.002, min_strength=2):
        self.cluster_pct = cluster_pct  # 0.2% clustering
        self.min_strength = min_strength
    
    def find_swing_points(self, df, lookbacks=[10, 20, 50]):
        """Find swing highs and lows at multiple timeframes"""
        highs = []
        lows = []
        
        for lb in lookbacks:
            for i in range(lb, len(df) - lb):
                # Swing High
                if df["h"].iloc[i] == df["h"].iloc[i-lb:i+lb+1].max():
                    highs.append({"idx": i, "price": df["h"].iloc[i], "lookback": lb})
                # Swing Low
                if df["l"].iloc[i] == df["l"].iloc[i-lb:i+lb+1].min():
                    lows.append({"idx": i, "price": df["l"].iloc[i], "lookback": lb})
        
        return highs, lows
    
    def find_equal_levels(self, prices, tolerance=0.001):
        """Find equal highs/lows (0.1% tolerance)"""
        equal_groups = []
        used = set()
        
        sorted_prices = sorted(prices, key=lambda x: x["price"])
        
        for i, p in enumerate(sorted_prices):
            if i in used:
                continue
            group = [p]
            for j in range(i+1, len(sorted_prices)):
                if j in used:
                    continue
                if abs(sorted_prices[j]["price"] - p["price"]) / p["price"] < tolerance:
                    group.append(sorted_prices[j])
                    used.add(j)
                else:
                    break
            if len(group) >= 2:
                equal_groups.append(group)
                used.add(i)
        
        return equal_groups
    
    def cluster_levels(self, levels):
        """Cluster nearby levels (0.2% tolerance)"""
        if not levels:
            return []
        
        sorted_levels = sorted(levels, key=lambda x: x["price"])
        clusters = []
        current_cluster = [sorted_levels[0]]
        
        for i in range(1, len(sorted_levels)):
            if abs(sorted_levels[i]["price"] - current_cluster[0]["price"]) / current_cluster[0]["price"] < self.cluster_pct:
                current_cluster.append(sorted_levels[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [sorted_levels[i]]
        
        clusters.append(current_cluster)
        
        # Score by strength (number of touches)
        result = []
        for cluster in clusters:
            avg_price = np.mean([l["price"] for l in cluster])
            strength = len(cluster)
            result.append({"price": avg_price, "strength": strength, "levels": cluster})
        
        return sorted(result, key=lambda x: x["strength"], reverse=True)
    
    def find_pdhl(self, df):
        """Find Previous Day High/Low"""
        df_daily = df.set_index("ts").resample("1D").agg({
            "o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"
        }).dropna()
        
        if len(df_daily) < 2:
            return None, None
        
        pdh = df_daily["h"].iloc[-2]
        pdl = df_daily["l"].iloc[-2]
        
        return pdh, pdl
    
    def build_liquidity_map(self, df):
        """Build complete liquidity map"""
        # Find swing points
        swing_highs, swing_lows = self.find_swing_points(df)
        
        # Find equal levels
        eq_highs = self.find_equal_levels(swing_highs)
        eq_lows = self.find_equal_levels(swing_lows)
        
        # Cluster all levels
        all_highs = self.cluster_levels(swing_highs)
        all_lows = self.cluster_levels(swing_lows)
        
        # PDH/PDL
        pdh, pdl = self.find_pdhl(df)
        
        # Filter by min_strength
        strong_highs = [h for h in all_highs if h["strength"] >= self.min_strength]
        strong_lows = [l for l in all_lows if l["strength"] >= self.min_strength]
        
        return {
            "resistance": strong_highs,
            "support": strong_lows,
            "equal_highs": eq_highs,
            "equal_lows": eq_lows,
            "pdh": pdh,
            "pdl": pdl,
        }


class LiquiditySweepDetector:
    """
    Phase 2: Detect liquidity sweeps (bullish/bearish traps)
    Based on LTDA algorithm:
    - Wick penetrates level
    - Close back above/below level
    - Level not violated in last 10 candles
    - 2 confirmation candles after
    """
    
    def __init__(self, lookback_fresh=10, confirm_candles=2, 
                 wick_ratio=1.0, stop_buffer=0.002):
        self.lookback_fresh = lookback_fresh
        self.confirm_candles = confirm_candles
        self.wick_ratio = wick_ratio  # minimum wick/body ratio
        self.stop_buffer = stop_buffer  # 0.2% beyond wick
    
    def is_level_fresh(self, df, level_price, candle_idx, tolerance=0.001):
        """Check if level hasn't been violated in last N candles"""
        start = max(0, candle_idx - self.lookback_fresh)
        for i in range(start, candle_idx):
            if abs(df["l"].iloc[i] - level_price) / level_price < tolerance:
                return False  # Already swept recently
        return True
    
    def detect_sweeps(self, df, liquidity_map):
        """
        Detect all liquidity sweeps in the data
        Returns list of signals with entry/stop/target
        """
        signals = []
        
        # Combine all levels
        levels = []
        for h in liquidity_map["resistance"]:
            levels.append({"price": h["price"], "type": "RESISTANCE", "strength": h["strength"]})
        for l in liquidity_map["support"]:
            levels.append({"price": l["price"], "type": "SUPPORT", "strength": l["strength"]})
        
        if liquidity_map["pdh"]:
            levels.append({"price": liquidity_map["pdh"], "type": "RESISTANCE", "strength": 3})
        if liquidity_map["pdl"]:
            levels.append({"price": liquidity_map["pdl"], "type": "SUPPORT", "strength": 3})
        
        # Check each candle for sweeps
        for i in range(self.lookback_fresh + 1, len(df) - self.confirm_candles):
            candle = df.iloc[i]
            
            for level in levels:
                level_price = level["price"]
                
                # BULLISH SWEEP: Support swept, candle closes above
                if level["type"] == "SUPPORT":
                    # Condition 1: Wick below support
                    if candle["l"] < level_price:
                        # Condition 2: Close above support
                        if candle["c"] > level_price:
                            # Condition 3: Level is fresh
                            if self.is_level_fresh(df, level_price, i):
                                # Check wick ratio
                                body = abs(candle["c"] - candle["o"])
                                lower_wick = min(candle["o"], candle["c"]) - candle["l"]
                                
                                if body > 0 and lower_wick >= body * self.wick_ratio:
                                    # Condition 4: Confirmation candles
                                    confirmed = True
                                    for j in range(1, self.confirm_candles + 1):
                                        if i + j < len(df):
                                            if df["c"].iloc[i+j] < level_price:
                                                confirmed = False
                                                break
                                    
                                    if confirmed:
                                        # Calculate entry/stop/target
                                        entry = candle["c"]
                                        stop = candle["l"] - entry * self.stop_buffer
                                        risk = entry - stop
                                        target1 = entry + risk * 2
                                        target2 = entry + risk * 3
                                        
                                        signals.append({
                                            "type": "LONG",
                                            "entry": entry,
                                            "stop": stop,
                                            "target1": target1,
                                            "target2": target2,
                                            "risk_pct": risk / entry * 100,
                                            "rr": 2.0,
                                            "level": level_price,
                                            "level_type": "SUPPORT",
                                            "strength": level["strength"],
                                            "idx": i,
                                            "ts": df["ts"].iloc[i],
                                        })
                
                # BEARISH SWEEP: Resistance swept, candle closes below
                elif level["type"] == "RESISTANCE":
                    # Condition 1: Wick above resistance
                    if candle["h"] > level_price:
                        # Condition 2: Close below resistance
                        if candle["c"] < level_price:
                            # Condition 3: Level is fresh
                            if self.is_level_fresh(df, level_price, i):
                                # Check wick ratio
                                body = abs(candle["c"] - candle["o"])
                                upper_wick = candle["h"] - max(candle["o"], candle["c"])
                                
                                if body > 0 and upper_wick >= body * self.wick_ratio:
                                    # Condition 4: Confirmation candles
                                    confirmed = True
                                    for j in range(1, self.confirm_candles + 1):
                                        if i + j < len(df):
                                            if df["c"].iloc[i+j] > level_price:
                                                confirmed = False
                                                break
                                    
                                    if confirmed:
                                        entry = candle["c"]
                                        stop = candle["h"] + entry * self.stop_buffer
                                        risk = stop - entry
                                        target1 = entry - risk * 2
                                        target2 = entry - risk * 3
                                        
                                        signals.append({
                                            "type": "SHORT",
                                            "entry": entry,
                                            "stop": stop,
                                            "target1": target1,
                                            "target2": target2,
                                            "risk_pct": risk / entry * 100,
                                            "rr": 2.0,
                                            "level": level_price,
                                            "level_type": "RESISTANCE",
                                            "strength": level["strength"],
                                            "idx": i,
                                            "ts": df["ts"].iloc[i],
                                        })
        
        return signals


class LiquiditySweepBacktest:
    """
    Phase 3: Backtest the liquidity sweep strategy
    """
    
    def __init__(self, capital=10.0, risk_per_trade=0.01, max_trades=50,
                 min_candle_gap=5, time_exit=50):
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.max_trades = max_trades
        self.min_candle_gap = min_candle_gap
        self.time_exit = time_exit
    
    def run(self, df, signals):
        """Run backtest on signals"""
        trades = []
        last_trade_idx = -self.min_candle_gap
        equity = self.capital
        
        for signal in signals:
            if len(trades) >= self.max_trades:
                break
            
            # Check minimum gap between trades
            if signal["idx"] - last_trade_idx < self.min_candle_gap:
                continue
            
            # Skip if too close to end of data
            if signal["idx"] + self.time_exit >= len(df):
                continue
            
            # Simulate trade
            entry = signal["entry"]
            stop = signal["stop"]
            target1 = signal["target1"]
            target2 = signal["target2"]
            
            risk_amount = equity * self.risk_per_trade
            position_size = risk_amount / abs(entry - stop) if abs(entry - stop) > 0 else 0
            
            # Check each candle after entry
            exit_price = None
            exit_reason = None
            
            for j in range(signal["idx"] + 1, min(signal["idx"] + self.time_exit, len(df))):
                high = df["h"].iloc[j]
                low = df["l"].iloc[j]
                close = df["c"].iloc[j]
                
                if signal["type"] == "LONG":
                    # Check stop loss
                    if low <= stop:
                        exit_price = stop
                        exit_reason = "SL"
                        break
                    # Check target
                    if high >= target2:
                        exit_price = target2
                        exit_reason = "TP2"
                        break
                    if high >= target1:
                        # Move stop to breakeven
                        stop = entry
                        exit_price = target1
                        exit_reason = "TP1"
                        break
                else:  # SHORT
                    # Check stop loss
                    if high >= stop:
                        exit_price = stop
                        exit_reason = "SL"
                        break
                    # Check target
                    if low <= target2:
                        exit_price = target2
                        exit_reason = "TP2"
                        break
                    if low <= target1:
                        stop = entry
                        exit_price = target1
                        exit_reason = "TP1"
                        break
            
            if exit_price is None:
                # Time exit
                exit_price = df["c"].iloc[min(signal["idx"] + self.time_exit, len(df)-1)]
                exit_reason = "TIME"
            
            # Calculate P&L
            if signal["type"] == "LONG":
                pnl_pct = (exit_price - entry) / entry
            else:
                pnl_pct = (entry - exit_price) / entry
            
            pnl = pnl_pct * position_size * entry  # raw P&L
            risk_reward = pnl / risk_amount if risk_amount > 0 else 0
            
            trades.append({
                "type": signal["type"],
                "entry": entry,
                "exit": exit_price,
                "stop": signal["stop"],
                "target1": target1,
                "level": signal["level"],
                "strength": signal["strength"],
                "pnl_pct": pnl_pct,
                "pnl_dollar": pnl,
                "risk_reward": risk_reward,
                "exit_reason": exit_reason,
                "idx": signal["idx"],
                "ts": signal["ts"],
            })
            
            equity += pnl
            last_trade_idx = signal["idx"]
        
        return self._compute_stats(trades, equity)
    
    def _compute_stats(self, trades, final_equity):
        """Compute backtest statistics"""
        if not trades:
            return {"total": 0, "accuracy": 0, "pf": 0, "pnl": 0, "trades": []}
        
        t = pd.DataFrame(trades)
        
        total = len(t)
        winners = t[t["pnl_dollar"] > 0]
        losers = t[t["pnl_dollar"] <= 0]
        
        win_rate = len(winners) / total * 100
        avg_win = winners["pnl_dollar"].mean() if len(winners) > 0 else 0
        avg_loss = losers["pnl_dollar"].mean() if len(losers) > 0 else 0
        
        gross_profit = winners["pnl_dollar"].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers["pnl_dollar"].sum()) if len(losers) > 0 else 0.01
        profit_factor = gross_profit / gross_loss
        
        total_pnl = t["pnl_dollar"].sum()
        total_return = (final_equity / self.capital - 1) * 100
        
        # Max drawdown
        equity_curve = [self.capital]
        for pnl in t["pnl_dollar"]:
            equity_curve.append(equity_curve[-1] + pnl)
        equity_curve = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_curve)
        dd = (peak - equity_curve) / peak
        max_dd = dd.max() * 100
        
        # Average R
        avg_r = t["risk_reward"].mean()
        
        # Long/Short breakdown
        longs = t[t["type"] == "LONG"]
        shorts = t[t["type"] == "SHORT"]
        
        long_wr = len(longs[longs["pnl_dollar"] > 0]) / len(longs) * 100 if len(longs) > 0 else 0
        short_wr = len(shorts[shorts["pnl_dollar"] > 0]) / len(shorts) * 100 if len(shorts) > 0 else 0
        
        return {
            "total": total,
            "winners": len(winners),
            "losers": len(losers),
            "accuracy": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "total_pnl": total_pnl,
            "total_return": total_return,
            "final_equity": final_equity,
            "max_drawdown": max_dd,
            "avg_r": avg_r,
            "long_trades": len(longs),
            "short_trades": len(shorts),
            "long_wr": long_wr,
            "short_wr": short_wr,
            "trades": trades,
        }
