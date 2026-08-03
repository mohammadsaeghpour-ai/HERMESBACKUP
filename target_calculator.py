"""
Advanced Target Calculation Engine for Crypto Trading
=====================================================
Implements: Fibonacci extensions, Measured moves, Pivot points,
ATR-based targets, Volume profile levels, Market structure targets.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict


# ── Constants ─────────────────────────────────────────────────────
FIB_EXTENSIONS = [0.618, 1.0, 1.272, 1.618, 2.0, 2.618, 3.618]
FIB_RETRACEMENTS = [0.236, 0.382, 0.500, 0.618, 0.786]
PHI = (1 + math.sqrt(5)) / 2  # 1.6180339...


# ── Data Classes ──────────────────────────────────────────────────
@dataclass
class Level:
    price: float
    label: str
    score: int = 0
    method: str = ""

    def __repr__(self):
        return f"{self.label}: ${self.price:,.2f} (score={self.score}, via {self.method})"


@dataclass
class TradeSetup:
    direction: str  # "long" or "short"
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    risk_reward: float
    position_size_btc: float = 0.0
    confluence_score: int = 0
    notes: List[str] = field(default_factory=list)

    def __repr__(self):
        rr_str = f"{self.risk_reward:.1f}:1"
        return (
            f"\n{'='*60}\n"
            f"  TRADE SETUP: {self.direction.upper()}\n"
            f"{'='*60}\n"
            f"  Entry:    ${self.entry:,.2f}\n"
            f"  Stop:     ${self.stop_loss:,.2f}\n"
            f"  TP1:      ${self.tp1:,.2f}  (R:R={abs(self.tp1 - self.entry)/abs(self.entry - self.stop_loss):.1f}:1)\n"
            f"  TP2:      ${self.tp2:,.2f}  (R:R={abs(self.tp2 - self.entry)/abs(self.entry - self.stop_loss):.1f}:1)\n"
            f"  TP3:      ${self.tp3:,.2f}  (R:R={abs(self.tp3 - self.entry)/abs(self.entry - self.stop_loss):.1f}:1)\n"
            f"  R:R:      {rr_str}\n"
            f"  Position: {self.position_size_btc:.4f} BTC\n"
            f"  Score:    {self.confluence_score}/5\n"
            f"  Notes:    {'; '.join(self.notes)}\n"
            f"{'='*60}"
        )


# ── 1. FIBONACCI METHODS ─────────────────────────────────────────
class FibonacciCalculator:
    """Fibonacci extensions, retracements, and projections."""

    @staticmethod
    def extensions(x: float, a: float, b: float) -> List[Level]:
        """
        Three-point extension: X(impulse start) → A(impulse end) → B(retrace end)
        Target = B + (A - X) × multiplier (long)
        """
        impulse = a - x
        return [
            Level(
                price=round(b + impulse * mult, 2),
                label=f"Fib Ext {mult:.3f}",
                method="fib_extension"
            )
            for mult in FIB_EXTENSIONS
        ]

    @staticmethod
    def retracement(low: float, high: float, direction: str = "long") -> List[Level]:
        """
        Fibonacci retracement levels for entry zones.
        Long: entry at retracement from low to high.
        """
        rng = high - low
        if direction == "long":
            return [
                Level(
                    price=round(low + rng * pct, 2),
                    label=f"Fib Ret {pct:.3f}",
                    method="fib_retracement"
                )
                for pct in FIB_RETRACEMENTS
            ]
        else:
            return [
                Level(
                    price=round(high - rng * pct, 2),
                    label=f"Fib Ret {pct:.3f}",
                    method="fib_retracement"
                )
                for pct in FIB_RETRACEMENTS
            ]

    @staticmethod
    def abcd(a: float, b: float, c: float, ratio: float = 1.0) -> Level:
        """
        AB=CD measured move: D = C + (B - A) × ratio
        Classic: ratio=1.0, Extended: ratio=1.618
        """
        ab = b - a
        return Level(
            price=round(c + ab * ratio, 2),
            label=f"AB=CD ({ratio:.3f})",
            method="abcd"
        )

    @staticmethod
    def harmonic_butterfly(x: float, a: float, b: float) -> dict:
        """Butterfly pattern: B=0.786XA, D=1.272XA."""
        xa = a - x
        b_level = round(x + xa * 0.786, 2)  # Expected B
        d_level = round(a + xa * 1.272, 2)   # Expected D
        return {"expected_B": b_level, "D_target": d_level}


# ── 2. PIVOT POINT METHODS ──────────────────────────────────────
class PivotCalculator:
    """Standard, Woodie, Camarilla, and DeMark pivots."""

    @staticmethod
    def standard(high: float, low: float, close: float) -> List[Level]:
        pp = (high + low + close) / 3
        return [
            Level(price=round(pp, 2),                    label="PP",   method="std_pivot"),
            Level(price=round(2*pp - low, 2),            label="R1",   method="std_pivot"),
            Level(price=round(pp + (high - low), 2),     label="R2",   method="std_pivot"),
            Level(price=round(high + 2*(pp - low), 2),   label="R3",   method="std_pivot"),
            Level(price=round(2*pp - high, 2),           label="S1",   method="std_pivot"),
            Level(price=round(pp - (high - low), 2),     label="S2",   method="std_pivot"),
            Level(price=round(low - 2*(high - pp), 2),   label="S3",   method="std_pivot"),
        ]

    @staticmethod
    def woodie(high: float, low: float, close: float) -> List[Level]:
        pp = (high + low + 2*close) / 4
        return [
            Level(price=round(pp, 2),                label="PP", method="woodie"),
            Level(price=round(2*pp - low, 2),        label="R1", method="woodie"),
            Level(price=round(pp + (high - low), 2), label="R2", method="woodie"),
            Level(price=round(2*pp - high, 2),       label="S1", method="woodie"),
            Level(price=round(pp - (high - low), 2), label="S2", method="woodie"),
        ]

    @staticmethod
    def camarilla(high: float, low: float, close: float) -> List[Level]:
        rng = high - low
        return [
            Level(price=round(close + rng * 1.1 / 2,  2), label="R4", method="camarilla"),
            Level(price=round(close + rng * 1.1 / 4,  2), label="R3", method="camarilla"),
            Level(price=round(close + rng * 1.1 / 6,  2), label="R2", method="camarilla"),
            Level(price=round(close + rng * 1.1 / 12, 2), label="R1", method="camarilla"),
            Level(price=round(close - rng * 1.1 / 12, 2), label="S1", method="camarilla"),
            Level(price=round(close - rng * 1.1 / 6,  2), label="S2", method="camarilla"),
            Level(price=round(close - rng * 1.1 / 4,  2), label="S3", method="camarilla"),
            Level(price=round(close - rng * 1.1 / 2,  2), label="S4", method="camarilla"),
        ]

    @staticmethod
    def demark(high: float, low: float, close: float, open_: float) -> List[Level]:
        if close < open_:
            x = high + 2*low + close
        elif close > open_:
            x = 2*high + low + close
        else:
            x = high + low + 2*close
        return [
            Level(price=round(x / 4, 2),       label="PP", method="demark"),
            Level(price=round(x / 2 - low, 2),  label="R1", method="demark"),
            Level(price=round(x / 2 - high, 2), label="S1", method="demark"),
        ]


# ── 3. ATR-BASED TARGETS ────────────────────────────────────────
class ATRCalculator:
    """ATR-based SL/TP, Chandelier exits, Keltner channels, sizing."""

    @staticmethod
    def atr_targets(
        entry: float,
        atr: float,
        direction: str = "long",
        sl_mult: float = 1.5,
        tp_multipliers: List[float] = [2.0, 3.0, 5.0],
    ) -> dict:
        sign = 1 if direction == "long" else -1
        sl = entry + sign * (-sl_mult * atr)
        tps = [entry + sign * m * atr for m in tp_multipliers]
        return {
            "sl": round(sl, 2),
            "tp1": round(tps[0], 2),
            "tp2": round(tps[1], 2),
            "tp3": round(tps[2], 2),
        }

    @staticmethod
    def chandelier_exit(
        highest_high: float,
        lowest_low: float,
        atr: float,
        direction: str = "long",
        mult: float = 3.0,
    ) -> float:
        if direction == "long":
            return round(highest_high - atr * mult, 2)
        else:
            return round(lowest_low + atr * mult, 2)

    @staticmethod
    def keltner_channel(
        ema20: float, atr10: float, mult: float = 1.5
    ) -> dict:
        return {
            "upper": round(ema20 + atr10 * mult, 2),
            "middle": round(ema20, 2),
            "lower": round(ema20 - atr10 * mult, 2),
        }

    @staticmethod
    def position_size(
        account_usd: float,
        risk_pct: float,
        atr: float,
        sl_mult: float = 1.5,
    ) -> float:
        """Returns BTC position size."""
        risk_usd = account_usd * risk_pct
        return round(risk_usd / (atr * sl_mult), 4)

    @staticmethod
    def atr_scale(atr_h1: float, target_tf: str) -> float:
        """Scale ATR to different timeframes.
        Note: sqrt scaling works well up to daily; weekly capped for crypto
        because crypto volatility doesn't scale linearly at long horizons.
        """
        factors = {"1H": 1, "4H": math.sqrt(4), "1D": math.sqrt(24), "1W": math.sqrt(168)}
        raw = atr_h1 * factors.get(target_tf, 1)
        # Cap weekly ATR to avoid unrealistic extrapolation in crypto
        if target_tf == "1W":
            raw = min(raw, atr_h1 * 12)  # Max 12× 1H ATR for weekly
        return round(raw, 2)


# ── 4. VOLUME PROFILE LEVELS ────────────────────────────────────
class VolumeProfileCalculator:
    """Calculate value area, POC, HVN/LVN from price-volume data."""

    @staticmethod
    def calculate(
        price_volumes: List[Tuple[float, float]],  # [(price, volume), ...]
        value_area_pct: float = 0.70,
    ) -> dict:
        """
        price_volumes: list of (price, volume) for each price level.
        Returns: POC, VAH, VAL, HVN, LVN.
        """
        if not price_volumes:
            return {}

        # Sort by volume descending for POC
        sorted_by_vol = sorted(price_volumes, key=lambda x: x[1], reverse=True)
        poc = sorted_by_vol[0][0]

        # Value Area: accumulate from POC outward until 70% of total volume
        total_vol = sum(v for _, v in price_volumes)
        target_vol = total_vol * value_area_pct
        accumulated = sorted_by_vol[0][1]

        # Sort prices for range
        prices_sorted = sorted(price_volumes, key=lambda x: x[0])
        poc_idx = next(
            i for i, (p, _) in enumerate(prices_sorted) if p == poc
        )

        va_low_idx = poc_idx
        va_high_idx = poc_idx

        while accumulated < target_vol and (va_low_idx > 0 or va_high_idx < len(prices_sorted) - 1):
            # Expand to whichever side has more volume
            vol_below = prices_sorted[va_low_idx - 1][1] if va_low_idx > 0 else 0
            vol_above = prices_sorted[va_high_idx + 1][1] if va_high_idx < len(prices_sorted) - 1 else 0

            if vol_below >= vol_above and va_low_idx > 0:
                va_low_idx -= 1
                accumulated += vol_below
            elif va_high_idx < len(prices_sorted) - 1:
                va_high_idx += 1
                accumulated += vol_above
            else:
                break

        # HVN/LVN classification (median volume as threshold)
        all_vols = [v for _, v in price_volumes]
        median_vol = sorted(all_vols)[len(all_vols) // 2]
        hvn_levels = [p for p, v in price_volumes if v > median_vol * 1.5]
        lvn_levels = [p for p, v in price_volumes if v < median_vol * 0.5]

        return {
            "poc": poc,
            "vah": prices_sorted[va_high_idx][0],
            "val": prices_sorted[va_low_idx][0],
            "hvn_levels": hvn_levels[:5],  # Top 5
            "lvn_levels": lvn_levels[:5],  # Top 5
            "total_volume": total_vol,
        }


# ── 5. MARKET STRUCTURE TARGETS ─────────────────────────────────
class MarketStructureCalculator:
    """BOS, CHoCH, Order Block, Liquidity, FVG targets."""

    @staticmethod
    def bos_target(
        bos_price: float,
        structure_start: float,
        atr: float,
        direction: str = "long",
    ) -> Level:
        """Break of Structure target projection."""
        dist = abs(bos_price - structure_start)
        sign = 1 if direction == "long" else -1
        target = bos_price + sign * dist
        return Level(price=round(target, 2), label="BOS Target", method="market_structure")

    @staticmethod
    def choch_entry(
        choch_price: float,
        trigger_level: float,
        atr: float,
        direction: str = "long",
    ) -> dict:
        """
        CHoCH + BOS entry setup.
        SL beyond the trigger swing, TP = 1.0-1.618× structure distance.
        """
        dist = abs(choch_price - trigger_level)
        sl = trigger_level if direction == "long" else trigger_level
        sign = 1 if direction == "long" else -1
        return {
            "entry": round(choch_price, 2),
            "sl": round(trigger_level - sign * 0.25 * atr, 2),
            "tp_1.0": round(choch_price + sign * dist * 1.0, 2),
            "tp_1.272": round(choch_price + sign * dist * 1.272, 2),
            "tp_1.618": round(choch_price + sign * dist * 1.618, 2),
        }

    @staticmethod
    def order_block(
        ob_high: float,
        ob_low: float,
        atr: float,
        direction: str = "long",
        buffer: float = 0.25,
    ) -> dict:
        """Entry from Order Block midpoint."""
        midpoint = (ob_high + ob_low) / 2
        sign = 1 if direction == "long" else -1
        return {
            "entry": round(midpoint, 2),
            "sl": round(ob_low - sign * buffer * atr, 2) if direction == "long"
                  else round(ob_high + sign * buffer * atr, 2),
            "ob_range": round(ob_high - ob_low, 2),
        }

    @staticmethod
    def fair_value_gap(
        gap_high: float, gap_low: float, direction: str = "long"
    ) -> dict:
        """FVG fill targets."""
        midpoint = (gap_high + gap_low) / 2
        return {
            "gap_top": gap_high,
            "gap_bottom": gap_low,
            "midpoint": round(midpoint, 2),
            "fill_target": gap_high if direction == "long" else gap_low,
        }

    @staticmethod
    def equal_hands_target(
        eq_level: float, atr: float, direction: str = "long"
    ) -> Level:
        """Target beyond equal highs/lows (liquidity sweep)."""
        sign = 1 if direction == "long" else -1
        target = eq_level + sign * 0.75 * atr
        return Level(price=round(target, 2), label="Liquidity Target", method="liquidity")


# ── 6. CONFLUENCE SCORING ────────────────────────────────────────
class ConfluenceScorer:
    """Score targets by confluence of multiple methods."""

    @staticmethod
    def score_target(
        target_price: float,
        fib_ext_levels: List[Level],
        pivot_levels: List[Level],
        volume_levels: dict,
        structure_levels: List[Level],
        measured_moves: List[Level],
        atr: float,
        tolerance_pct: float = 0.003,  # 0.3% tolerance
    ) -> Tuple[int, List[str]]:
        """
        Score a target price by how many methods confirm it.
        Returns score 0-5.
        """
        score = 0
        methods_found = []

        # Check Fibonacci
        for lv in fib_ext_levels:
            if abs(lv.price - target_price) / target_price < tolerance_pct:
                score += 1
                methods_found.append(f"Fib({lv.label})")
                break

        # Check Pivot points
        for lv in pivot_levels:
            if abs(lv.price - target_price) / target_price < tolerance_pct:
                score += 1
                methods_found.append(f"Pivot({lv.label})")
                break

        # Check Volume profile
        for key in ["poc", "vah", "val"]:
            if key in volume_levels and abs(volume_levels[key] - target_price) / target_price < tolerance_pct:
                score += 1
                methods_found.append(f"VP({key.upper()})")
                break

        # Check Market structure
        for lv in structure_levels:
            if abs(lv.price - target_price) / target_price < tolerance_pct:
                score += 1
                methods_found.append(f"MS({lv.label})")
                break

        # Check Measured moves
        for lv in measured_moves:
            if abs(lv.price - target_price) / target_price < tolerance_pct:
                score += 1
                methods_found.append(f"MM({lv.label})")
                break

        return score, methods_found


# ── 7. MASTER TRADE CALCULATOR ───────────────────────────────────
class TradeEngine:
    """Combines all methods into a single trade setup."""

    def __init__(
        self,
        current_price: float,
        atr_1h: float,
        high_1h: float,
        low_1h: float,
        close_1h: float,
        open_1h: float,
        swing_high: float,
        swing_low: float,
        retrace_level: float = 0.618,
        account_usd: float = 100_000,
        risk_pct: float = 0.01,
    ):
        self.price = current_price
        self.atr = atr_1h
        self.high = high_1h
        self.low = low_1h
        self.close = close_1h
        self.open = open_1h
        self.swing_high = swing_high
        self.swing_low = swing_low
        self.retrace = retrace_level
        self.account = account_usd
        self.risk = risk_pct

        self.fib = FibonacciCalculator()
        self.pivots = PivotCalculator()
        self.atr_calc = ATRCalculator()
        self.vp = VolumeProfileCalculator()
        self.ms = MarketStructureCalculator()
        self.scorer = ConfluenceScorer()

    def compute_long_setup(self) -> TradeSetup:
        """Full long trade setup calculation."""
        notes = []

        # 1. Fibonacci retracement entry
        retrace_entry = self.swing_low + (self.swing_high - self.swing_low) * self.retrace
        notes.append(f"Fib retrace entry ({self.retrace}): ${retrace_entry:,.2f}")

        # 2. Fibonacci extensions
        fib_exts = self.fib.extensions(self.swing_low, self.swing_high, retrace_entry)
        ext_1618 = next(lv for lv in fib_exts if "1.618" in lv.label)
        ext_1272 = next(lv for lv in fib_exts if "1.272" in lv.label)
        ext_2000 = next(lv for lv in fib_exts if "2.000" in lv.label)

        # 3. Pivot points (standard)
        std_pivots = self.pivots.standard(self.high, self.low, self.close)
        pp = next(lv for lv in std_pivots if lv.label == "PP")
        r1 = next(lv for lv in std_pivots if lv.label == "R1")
        r2 = next(lv for lv in std_pivots if lv.label == "R2")

        # 4. ATR-based levels
        atr_targets = self.atr_calc.atr_targets(retrace_entry, self.atr, "long")

        # 5. AB=CD measured move
        abcd = self.fib.abcd(self.swing_low, self.swing_high, retrace_entry, 1.618)

        # 6. Entry, SL, TP
        entry = retrace_entry
        sl = atr_targets["sl"]

        # Find best TP1, TP2, TP3 by confluence
        all_candidates = [ext_1272, ext_1618, ext_2000, pp, r1, r2, abcd]
        candidates_for_tp = [c for c in all_candidates if c.price > entry]

        # Sort by proximity, pick top 3
        candidates_for_tp.sort(key=lambda x: x.price)
        tp1 = candidates_for_tp[0].price if len(candidates_for_tp) > 0 else atr_targets["tp1"]
        tp2 = candidates_for_tp[1].price if len(candidates_for_tp) > 1 else atr_targets["tp2"]
        tp3 = candidates_for_tp[2].price if len(candidates_for_tp) > 2 else atr_targets["tp3"]

        # Position sizing
        pos_size = self.atr_calc.position_size(self.account, self.risk, self.atr)

        # Risk:Reward
        risk = entry - sl
        reward = tp2 - entry
        rr = reward / risk if risk > 0 else 0

        return TradeSetup(
            direction="long",
            entry=round(entry, 2),
            stop_loss=round(sl, 2),
            tp1=round(tp1, 2),
            tp2=round(tp2, 2),
            tp3=round(tp3, 2),
            risk_reward=round(rr, 2),
            position_size_btc=pos_size,
            confluence_score=0,
            notes=notes,
        )

    def compute_short_setup(self) -> TradeSetup:
        """Full short trade setup calculation."""
        notes = []

        # Fibonacci retrace for short entry
        retrace_entry = self.swing_high - (self.swing_high - self.swing_low) * self.retrace
        notes.append(f"Fib retrace entry ({self.retrace}): ${retrace_entry:,.2f}")

        # Extensions downward
        impulse = self.swing_high - self.swing_low
        ext_1272 = retrace_entry - impulse * 1.272
        ext_1618 = retrace_entry - impulse * 1.618
        ext_2000 = retrace_entry - impulse * 2.000

        # Pivots
        std_pivots = self.pivots.standard(self.high, self.low, self.close)
        s1 = next(lv for lv in std_pivots if lv.label == "S1")
        s2 = next(lv for lv in std_pivots if lv.label == "S2")

        # ATR targets
        atr_t = self.atr_calc.atr_targets(retrace_entry, self.atr, "short")

        entry = retrace_entry
        sl = atr_t["sl"]

        candidates = sorted([ext_1272, ext_1618, ext_2000, s1.price, s2.price])
        candidates_below = [c for c in candidates if c < entry]
        candidates_below.sort(reverse=True)

        tp1 = candidates_below[0] if len(candidates_below) > 0 else atr_t["tp1"]
        tp2 = candidates_below[1] if len(candidates_below) > 1 else atr_t["tp2"]
        tp3 = candidates_below[2] if len(candidates_below) > 2 else atr_t["tp3"]

        pos_size = self.atr_calc.position_size(self.account, self.risk, self.atr)
        risk = sl - entry
        reward = entry - tp2
        rr = reward / risk if risk > 0 else 0

        return TradeSetup(
            direction="short",
            entry=round(entry, 2),
            stop_loss=round(sl, 2),
            tp1=round(tp1, 2),
            tp2=round(tp2, 2),
            tp3=round(tp3, 2),
            risk_reward=round(rr, 2),
            position_size_btc=pos_size,
            confluence_score=0,
            notes=notes,
        )


# ══════════════════════════════════════════════════════════════════
# DEMO: Run all calculations with current BTC/ETH data
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("=" * 70)
    print("  ADVANCED TARGET CALCULATION ENGINE — DEMO WITH BTC/ETH DATA")
    print("=" * 70)

    # ── BTC Example ──
    print("\n" + "─" * 70)
    print("  BTC/USDT — H1 ANALYSIS")
    print("─" * 70)

    # Example market data (replace with live data)
    BTC_PRICE = 60_500
    BTC_ATR_1H = 177
    BTC_HIGH_4H = 61_200
    BTC_LOW_4H = 59_800
    BTC_CLOSE_4H = 60_600
    BTC_OPEN_4H = 60_100
    BTC_SWING_HIGH = 62_500
    BTC_SWING_LOW = 58_000

    engine = TradeEngine(
        current_price=BTC_PRICE,
        atr_1h=BTC_ATR_1H,
        high_1h=BTC_HIGH_4H,
        low_1h=BTC_LOW_4H,
        close_1h=BTC_CLOSE_4H,
        open_1h=BTC_OPEN_4H,
        swing_high=BTC_SWING_HIGH,
        swing_low=BTC_SWING_LOW,
        account_usd=100_000,
        risk_pct=0.01,
    )

    # Fibonacci
    fib = FibonacciCalculator()
    print("\n  [1] FIBONACCI EXTENSIONS (from swing $58,000→$62,500, retrace to $60,310)")
    exts = fib.extensions(58_000, 62_500, 60_310)
    for e in exts:
        print(f"      {e}")

    print("\n  [2] FIBONACCI RETRACEMENTS")
    retrace = fib.retracement(58_000, 62_500)
    for r in retrace:
        print(f"      {r}")

    print("\n  [3] AB=CD MEASURED MOVES")
    for ratio in [1.0, 1.272, 1.618]:
        d = fib.abcd(58_000, 62_500, 60_310, ratio)
        print(f"      {d}")

    # Pivots
    pivots = PivotCalculator()
    print("\n  [4] STANDARD PIVOTS (from 4H candle)")
    for p in pivots.standard(BTC_HIGH_4H, BTC_LOW_4H, BTC_CLOSE_4H):
        print(f"      {p}")

    print("\n  [5] WOODIE PIVOTS")
    for p in pivots.woodie(BTC_HIGH_4H, BTC_LOW_4H, BTC_CLOSE_4H):
        print(f"      {p}")

    print("\n  [6] CAMARILLA PIVOTS")
    for p in pivots.camarilla(BTC_HIGH_4H, BTC_LOW_4H, BTC_CLOSE_4H):
        print(f"      {p}")

    print("\n  [7] DEMARK PIVOTS")
    for p in pivots.demark(BTC_HIGH_4H, BTC_LOW_4H, BTC_CLOSE_4H, BTC_OPEN_4H):
        print(f"      {p}")

    # ATR
    atr = ATRCalculator()
    print("\n  [8] ATR-BASED TARGETS")
    print(f"      BTC 1H ATR = ${BTC_ATR_1H}")
    print(f"      4H ATR ≈ ${atr.atr_scale(BTC_ATR_1H, '4H')}")
    print(f"      Daily ATR ≈ ${atr.atr_scale(BTC_ATR_1H, '1D')}")
    print(f"      Weekly ATR ≈ ${atr.atr_scale(BTC_ATR_1H, '1W')}")
    print(f"      Position size (1% risk, $100K): {atr.position_size(100_000, 0.01, BTC_ATR_1H)} BTC")
    at = atr.atr_targets(60_500, BTC_ATR_1H, "long")
    print(f"      Long from $60,500: SL=${at['sl']}, TP1=${at['tp1']}, TP2=${at['tp2']}, TP3=${at['tp3']}")

    # Full trade setup
    print("\n  [9] FULL LONG TRADE SETUP")
    long_setup = engine.compute_long_setup()
    print(long_setup)

    print("\n  [10] FULL SHORT TRADE SETUP")
    short_setup = engine.compute_short_setup()
    print(short_setup)

    # ── ETH Example ──
    print("\n" + "─" * 70)
    print("  ETH/USDT — H1 ANALYSIS")
    print("─" * 70)

    ETH_ATR_1H = 8
    print(f"  ETH 1H ATR = ${ETH_ATR_1H}")
    print(f"  BTC/ETH ratio target: ${BTC_ATR_1H / ETH_ATR_1H:.2f}x ATR")
    print(f"  ETH conservative TP (2×ATR): ${ETH_ATR_1H * 2}")
    print(f"  ETH standard TP (3×ATR): ${ETH_ATR_1H * 3}")
    print(f"  ETH aggressive TP (5×ATR): ${ETH_ATR_1H * 5}")
    print(f"  ETH for $50 move: need {50 / ETH_ATR_1H:.1f}× ATR")
    print(f"  BTC for $1500 move: need {1500 / BTC_ATR_1H:.1f}× ATR")

    # Confluence scoring example
    print("\n  [11] CONFLUENCE SCORING EXAMPLE")
    print("  Testing target price $63,250...")
    score, methods = ConfluenceScorer.score_target(
        target_price=63_250,
        fib_ext_levels=exts,
        pivot_levels=pivots.standard(BTC_HIGH_4H, BTC_LOW_4H, BTC_CLOSE_4H),
        volume_levels={"poc": 60_600, "vah": 61_500, "val": 59_800},
        structure_levels=[],
        measured_moves=[fib.abcd(58_000, 62_500, 60_310, 1.618)],
        atr=BTC_ATR_1H,
    )
    print(f"  Score: {score}/5")
    print(f"  Methods: {', '.join(methods)}")

    print("\n" + "=" * 70)
    print("  ENGINE READY — Import TradeEngine for real-time calculations")
    print("=" * 70)
