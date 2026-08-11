"""
Advanced Acceleration Agent — Hidden Momentum Detection (OPTIMIZED)
===================================================================
Detects acceleration (change in velocity) across multiple timeframes:
1. **Price Velocity** — rate of price change
2. **Price Acceleration** — rate of change of velocity
3. **Volume Velocity** — rate of volume change
4. **Volume Acceleration** — rate of change of volume velocity
5. **Hidden Momentum** — acceleration in non-standard timeframes (8H, 12H)
6. **Time Compression** — squeeze detection from acceleration
7. **Acceleration Divergence** — price vs acceleration divergence

Weight: 1.2
Optimized: threshold=0.10-0.20, hold=8-15
"""
from hqip.agents.base import BaseAgent, AgentOutput
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


class AccelerationAgent(BaseAgent):
    """Multi-dimensional acceleration analysis for hidden momentum detection.

    Combines 7 acceleration subsystems to detect changes in market speed
    that are invisible in standard timeframes.

    Attributes
    ----------
    name : str
        Agent identifier.
    weight : float
        Consensus weight (1.2).
    """
    name = "Acceleration"
    weight = 1.2

    # ------------------------------------------------------------------ #
    #  Core: Price Velocity & Acceleration
    # ------------------------------------------------------------------ #
    @staticmethod
    def _price_velocity(close_arr, window=5):
        """Calculate price velocity (rate of change)."""
        if len(close_arr) < window + 1:
            return 0.0, 0.0
        velocity = np.mean(np.abs(np.diff(close_arr[-window-1:])))
        direction = np.mean(np.diff(close_arr[-window-1:]))
        return velocity, direction

    @staticmethod
    def _price_acceleration(close_arr, window=5):
        """Calculate price acceleration (rate of change of velocity)."""
        if len(close_arr) < window + 2:
            return 0.0
        changes = np.diff(close_arr)
        acceleration = np.mean(np.abs(np.diff(changes[-window-1:])))
        return acceleration

    # ------------------------------------------------------------------ #
    #  Core: Volume Velocity & Acceleration
    # ------------------------------------------------------------------ #
    @staticmethod
    def _volume_velocity(vol_arr, window=5):
        """Calculate volume velocity."""
        if len(vol_arr) < window + 1:
            return 0.0
        velocity = np.mean(np.abs(np.diff(vol_arr[-window-1:])))
        return velocity

    @staticmethod
    def _volume_acceleration(vol_arr, window=5):
        """Calculate volume acceleration."""
        if len(vol_arr) < window + 2:
            return 0.0
        changes = np.diff(vol_arr)
        acceleration = np.mean(np.abs(np.diff(changes[-window-1:])))
        return acceleration

    # ------------------------------------------------------------------ #
    #  Core: Hidden Momentum (non-standard timeframes)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _hidden_momentum(close_arr, vol_arr):
        """Detect momentum in non-standard timeframes.
        
        Hidden momentum = velocity * acceleration * volume_momentum
        """
        if len(close_arr) < 20:
            return 0.0
        
        v15 = np.mean(np.diff(close_arr[-16:]))
        v20 = np.mean(np.diff(close_arr[-21:]))
        hidden_accel = v15 - v20
        vol_mom = np.mean(vol_arr[-5:]) / max(np.mean(vol_arr[-20:]), 1)
        hidden_score = hidden_accel * vol_mom
        return hidden_score

    # ------------------------------------------------------------------ #
    #  Core: Time Compression (squeeze detection)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _time_compression(close_arr, window=20):
        """Detect time compression — when acceleration drops to near zero."""
        if len(close_arr) < window + 2:
            return 0.0
        
        changes = np.diff(close_arr)
        recent_accel = np.mean(np.abs(np.diff(changes[-10:])))
        older_accel = np.mean(np.abs(np.diff(changes[-20:-10])))
        
        if older_accel == 0:
            return 0.0
        
        compression = 1.0 - (recent_accel / older_accel)
        return compression

    # ------------------------------------------------------------------ #
    #  Core: Acceleration Divergence
    # ------------------------------------------------------------------ #
    @staticmethod
    def _acceleration_divergence(close_arr, window=20):
        """Detect divergence between price and acceleration."""
        if len(close_arr) < window + 2:
            return 0.0
        
        price_change = close_arr[-1] - close_arr[-window]
        changes = np.diff(close_arr)
        accel_change = np.mean(np.abs(np.diff(changes[-window:])))
        prev_accel_change = np.mean(np.abs(np.diff(changes[-2*window:-window])))
        
        if prev_accel_change == 0:
            return 0.0
        
        if price_change > 0 and accel_change < prev_accel_change:
            return -1.0
        elif price_change < 0 and accel_change > prev_accel_change:
            return 1.0
        
        return 0.0

    # ------------------------------------------------------------------ #
    #  Main Analysis (BaseAgent interface)
    # ------------------------------------------------------------------ #
    def analyze(self, df, **kwargs):
        """Run acceleration analysis.
        
        Parameters
        ----------
        df : dict
            Must contain 'close' and 'volume' arrays.
        **kwargs : dict
            Additional parameters.
        
        Returns
        -------
        AgentOutput
        """
        close = np.array(df.get('close', []), dtype=float)
        volume = np.array(df.get('volume', []), dtype=float)
        
        if len(close) < 25 or len(volume) < 25:
            return AgentOutput(
                agent_name=self.name,
                direction="NEUTRAL",
                confidence=0.0,
                score=0.0,
                evidence=["Insufficient data for acceleration analysis"],
                reasoning="Need at least 25 data points",
                weight=self.weight,
                error="insufficient_data"
            )
        
        # 1. Price Velocity
        vel, vel_dir = self._price_velocity(close)
        
        # 2. Price Acceleration
        accel = self._price_acceleration(close)
        
        # 3. Volume Velocity
        vol_vel = self._volume_velocity(volume)
        
        # 4. Volume Acceleration
        vol_accel = self._volume_acceleration(volume)
        
        # 5. Hidden Momentum
        hidden = self._hidden_momentum(close, volume)
        
        # 6. Time Compression
        compression = self._time_compression(close)
        
        # 7. Acceleration Divergence
        div = self._acceleration_divergence(close)
        
        # === Scoring (OPTIMIZED) ===
        score = 0.0
        reasons = []
        
        # Price Acceleration (35% weight)
        if accel > 0:
            if vel_dir > 0:
                score += 0.35
                reasons.append(f'Price accel UP ({accel:.4f})')
            else:
                score -= 0.35
                reasons.append(f'Price accel DOWN ({accel:.4f})')
        
        # Volume Momentum (25% weight)
        vol_mom = np.mean(volume[-5:]) / max(np.mean(volume[-20:]), 1)
        if vol_mom > 1.5:
            score += 0.25
            reasons.append(f'Volume accelerating ({vol_mom:.2f}x)')
        elif vol_mom < 0.5:
            score -= 0.25
            reasons.append(f'Volume decelerating ({vol_mom:.2f}x)')
        
        # Hidden Momentum (25% weight)
        if hidden > 0:
            score += 0.25
            reasons.append(f'Hidden momentum positive ({hidden:.4f})')
        elif hidden < 0:
            score -= 0.25
            reasons.append(f'Hidden momentum negative ({hidden:.4f})')
        
        # Time Compression (5% weight)
        if compression > 0.5:
            reasons.append(f'Time compression ({compression:.2f}) — breakout imminent')
        
        # Acceleration Divergence (10% weight)
        if div > 0:
            score += 0.10
            reasons.append('Bullish acceleration divergence')
        elif div < 0:
            score -= 0.10
            reasons.append('Bearish acceleration divergence')
        
        # Direction (OPTIMIZED: threshold 0.10-0.20)
        if score > 0.10:
            direction = "BUY"
        elif score < -0.10:
            direction = "SELL"
        else:
            direction = "NEUTRAL"
        
        # Confidence
        confirming = sum(1 for r in reasons if 'UP' in r or 'positive' in r or 'accelerating' in r)
        total_factors = len(reasons)
        confidence = (confirming / max(total_factors, 1)) * 100
        
        return AgentOutput(
            agent_name=self.name,
            direction=direction,
            confidence=confidence,
            score=score,
            evidence=reasons,
            reasoning=f"Acceleration analysis: vel={vel:.4f}, accel={accel:.4f}, hidden={hidden:.4f}, compression={compression:.2f}",
            data={
                'velocity': vel,
                'velocity_direction': vel_dir,
                'acceleration': accel,
                'volume_velocity': vol_vel,
                'volume_acceleration': vol_accel,
                'hidden_momentum': hidden,
                'time_compression': compression,
                'acceleration_divergence': div,
                'volume_momentum': vol_mom,
            },
            weight=self.weight,
        )
