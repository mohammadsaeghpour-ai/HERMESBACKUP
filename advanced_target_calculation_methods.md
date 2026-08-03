# Advanced Target Calculation Methods for Crypto Trading

## Reference Data
- BTC 1H ATR: $177
- ETH 1H ATR: $8
- BTC/ETH ratio: 0.0296
- Target moves: BTC $1500 / ETH $50

---

## 1. Fibonacci Extensions (1.272, 1.618, 2.0)

### Method: Three-Point Extension (XAB)
Given swing points: X (start), A (end of impulse), B (end of retracement)

```
Extension Level = B + (A - X) * multiplier   (long trade)
Extension Level = B - (X - A) * multiplier   (short trade)
```

**Key Multipliers:**
| Level | Extension | Typical Role |
|-------|-----------|-------------|
| 0.618 | 0.618 | Shallow target / first partial |
| 1.000 | 1.000 | Measured move equality |
| 1.272 | 1.272 | Conservative extension |
| 1.618 | 1.618 | Primary extension target |
| 2.000 | 2.000 | Aggressive extension |
| 2.618 | 2.618 | Blow-off / extreme target |

### Example: BTC Long Setup
```
X = $58,000 (swing low)
A = $60,500 (swing high)   → impulse = $2,500
B = $59,200 (retracement)

1.272 extension = $59,200 + $2,500 × 1.272 = $62,380
1.618 extension = $59,200 + $2,500 × 1.618 = $63,245
2.000 extension = $59,200 + $2,500 × 2.000 = $64,200

→ 1.618 target = $63,245 → profit = $4,045 (from entry at B)
```

### Fibonacci Retracement (for Entry)
```
Long entry: Low + (High - Low) × retrace_level
Short entry: High - (High - Low) × retrace_level

Key retrace levels: 0.382, 0.500, 0.618, 0.786
Optimal entry zone: 0.500 - 0.618 retrace
```

### Fibonacci Time Zones (confluence signal)
```
Zone n = swing_low_date + F(n) × time_unit
F(n) = Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13, 21, 34...
```

---

## 2. Measured Moves / AB=CD Pattern

### Classic AB=CD
```
Leg A→B should ≈ Leg C→D in price AND time

D = C + (B - A)              [price projection]
Time(C→D) ≈ Time(A→B)        [time symmetry]
```

### Extended AB=CD (with ratios)
```
D = C + (B - A) × ratio

Common ratios:
  1.000  - equality (classic AB=CD)
  1.272  - compressed continuation
  1.618  - extended continuation (most reliable)
  2.618  - extended trend continuation
```

### Example: BTC AB=CD Long
```
A = $57,500 (initial low)
B = $59,800 (first high)     → AB = +$2,300
C = $58,600 (pullback low)

Classic: D = $58,600 + $2,300 = $60,900
1.618:   D = $58,600 + $2,300 × 1.618 = $62,321
```

### Three-Drive Pattern (AB=CD squared)
```
Drive 1: A→B (impulse)
Retrace 1: B→C (0.618 retrace of A→B)
Drive 2: C→D (1.272 or 1.618 extension of B→C)
Retrace 2: D→E (0.618 retrace of C→D)
Drive 3: E→F = 1.272 or 1.618 extension of D→E
```

### Butterfly Pattern (Harmonic Extension)
```
B = 0.786 retrace of XA
D = 1.272 extension of XA
  OR
D = AB measured from C:
  D = C + (B-A) × 1.618

Entry at D, SL below 1.414 extension of XA
TP1 = 0.382 retrace of AD (from D)
TP2 = 0.618 retrace of AD (from D)
```

---

## 3. Pivot Points

### Standard Pivots (Floor Traders)
```
Pivot (PP) = (High + Low + Close) / 3

R1 = 2×PP - Low
R2 = PP + (High - Low)
R3 = High + 2×(PP - Low)

S1 = 2×PP - High
S2 = PP - (High - Low)
S3 = Low - 2×(High - PP)
```

### Woodie Pivots
```
PP = (High + Low + 2×Close) / 4

R1 = 2×PP - Low
R2 = PP + (High - Low)
S1 = 2×PP - High
S2 = PP - (High - Low)
```

### Camarilla Pivots (Intraday Scalping)
```
Range = High - Low

R4 = Close + Range × 1.1 / 2
R3 = Close + Range × 1.1 / 4
R2 = Close + Range × 1.1 / 6
R1 = Close + Range × 1.1 / 12

S1 = Close - Range × 1.1 / 12
S2 = Close - Range × 1.1 / 6
S3 = Close - Range × 1.1 / 4
S4 = Close - Range × 1.1 / 12

Key: R3/S3 = breakout level, R4/S4 = extreme target
     R1/S1-R2/S2 = mean reversion zone
```

### DeMark Pivots (Conditional on Close)
```
IF Close < Open:  X = High + 2×Low + Close
IF Close > Open:  X = 2×High + Low + Close
IF Close = Open:  X = High + Low + 2×Close

PP = X / 4
R1 = X / 2 - Low
S1 = X / 2 - High
```

### Example: BTC Standard Pivots
```
Previous 4H candle: High=$61,200, Low=$59,800, Close=$60,600

PP  = ($61,200 + $59,800 + $60,600) / 3 = $60,533
R1  = 2×$60,533 - $59,800 = $61,267
R2  = $60,533 + ($61,200 - $59,800) = $61,933
R3  = $61,200 + 2×($60,533 - $59,800) = $62,667
S1  = 2×$60,533 - $61,200 = $59,867
S2  = $60,533 - ($61,200 - $59,800) = $59,133
S3  = $59,800 - 2×($60,600 - $60,533) = $59,667
```

---

## 4. ATR-Based Targets

### ATR Multiplier Method
```
SL  = Entry ± ATR × SL_multiplier
TP1 = Entry ± ATR × TP1_multiplier
TP2 = Entry ± ATR × TP2_multiplier
TP3 = Entry ± ATR × TP3_multiplier

Standard multipliers:
  SL:  1.0-1.5× ATR (tight) / 2.0-2.5× ATR (standard)
  TP1: 2.0× ATR
  TP2: 3.0× ATR
  TP3: 4.5-5.0× ATR
```

### Chandelier Exit (Trailing Stop)
```
Long:  Stop = Highest_High(N) - ATR(N) × multiplier
Short: Stop = Lowest_Low(N) + ATR(N) × multiplier

Standard: N=22, multiplier=3.0
```

### Keltner Channel Targets
```
Middle = EMA(Close, 20)
Upper  = EMA(Close, 20) + ATR(10) × 1.5
Lower  = EMA(Close, 20) - ATR(10) × 1.5

Targets: Touch opposite band = full move
         Breakout band entry with SL at middle band
```

### Volatility-Adjusted Position Sizing
```
Risk per trade = Account × Risk%
Position size  = Risk_amount / (ATR × SL_multiplier)

Example: $100K account, 1% risk, BTC:
  Risk = $1,000
  Position = $1,000 / ($177 × 1.5) = 3.76 BTC ≈ 3.7 BTC
```

### ATR Scaling for Timeframes
```
ATR_H1 × sqrt(4)  = ATR_4H approximation
ATR_H1 × sqrt(24) = ATR_D1 approximation
ATR_H1 × sqrt(168)= ATR_W1 approximation (7-day)

BTC 1H ATR=$177:
  4H ATR ≈ $354
  Daily ATR ≈ $867
  Weekly ATR ≈ $725 (cap for crypto)
```

### Specific ATR Targets with Current Data
```
BTC 1H ATR = $177
  Conservative TP (2×ATR):  $354 move
  Standard TP (3×ATR):     $531 move
  Aggressive TP (5×ATR):   $885 move
  To reach $1500 target:    need ~8.5× ATR (multi-timeframe confluence needed)

ETH 1H ATR = $8
  Conservative TP (2×ATR):  $16 move
  Standard TP (3×ATR):     $24 move
  Aggressive TP (5×ATR):   $40 move
  To reach $50 target:      need ~6.25× ATR (multi-timeframe)
```

---

## 5. Volume Profile Support/Resistance

### Value Area (VA)
```
Value Area High (VAH): Upper boundary of 70% volume
Value Area Low (VAL):  Lower boundary of 70% volume
Point of Control (POC): Price level with highest volume

VA calculation:
1. Sort price levels by volume descending
2. Sum volumes from POC outward until 70% of total
3. Upper limit = VAH, Lower limit = VAL
```

### Volume Profile Fixed Range
```
For a defined range (e.g., last 100 candles):
  POC  = price with maximum volume
  VAH  = highest price containing 70% of total volume
  VAL  = lowest price containing 70% of total volume
  HVN  = High Volume Node (strong S/R)
  LVN  = Low Volume Node (price moves through quickly)
```

### Volume Profile Trading Rules
```
Long Setup:
  Entry:  Bounce off VAL or POC in uptrend
  SL:     Below VAL by 0.5× ATR
  TP1:    POC (if entered below POC)
  TP2:    VAH
  TP3:    Next HVN above VAH

Short Setup:
  Entry:  Rejection at VAH or POC in downtrend
  SL:     Above VAH by 0.5× ATR
  TP1:    POC (if entered above POC)
  TP2:    VAL
  TP3:    Next HVN below VAL

LVN zones = price vacuums (use for breakouts)
HVN zones = price magnets (use for mean reversion / entries)
```

### Developing Volume Profile (Real-time)
```
As new candles print, update running totals:
  For each price level p:
    vol(p) += volume at p

  POC = argmax(vol(p))
  VAH/VAL = 70% boundary recalculation every N bars
```

### Market Profile / TPO (Time Price Opportunity)
```
TPO count = number of 30-min periods price visited level
  → Similar to volume profile but measures time, not volume
  → More robust for low-volume crypto pairs

Single Print Zones = one TPO wide (breakout/breakaway gaps)
  → Price typically revisits to fill (reversion target)
```

---

## 6. Market Structure Targets

### Break of Structure (BOS) Targets
```
Uptrend BOS: Price breaks above previous swing high
  → Target: distance from last swing low to BOS point projected upward
  → TP = BOS_price + (BOS_price - last_swing_low)

Downtrend BOS: Price breaks below previous swing low
  → Target: distance from last swing high to BOS point projected downward
  → TP = BOS_price - (last_swing_high - BOS_price)
```

### Change of Character (CHoCH) + Target
```
CHoCH occurs when trend structure reverses:
  Bullish CHoCH: Lower low followed by break above previous swing high
  → Entry at BOS above structure
  → SL below the lower low that triggered the CHoCH
  → Target: 1.0-1.618× the distance of the structure break

  Bearish CHoCH: Higher high followed by break below previous swing low
  → Entry at BOS below structure
  → SL above the higher high that triggered the CHoCH
  → Target: 1.0-1.618× the distance of the structure break
```

### Order Block Targets
```
Order Block (OB): Last opposing candle before impulsive move
  Bullish OB: Last red candle before bullish impulse → demand zone
  Bearish OB: Last green candle before bearish impulse → supply zone

Entry: Limit order at OB midpoint
SL:    Beyond OB extreme by buffer (0.25× ATR)
TP1:   Previous swing (1:1 R:R)
TP2:   Opposing OB / liquidity zone
TP3:   1.618× OB-to-next-S/R distance
```

### Liquidity Targets (Stop Hunts / Smart Money)
```
Equal Highs/Lows = resting liquidity
  → Price sweeps equal highs/lows before reversing
  → Target: beyond the equal highs/lows by 0.5-1× ATR

Buy-side Liquidity (BSL): Stops above equal highs / swing highs
Sell-side Liquidity (SSL): Stops below equal lows / swing lows

Inducement zones: Minor S/R just before major S/R
  → Price takes inducement first, then reverses at real level
```

### Fair Value Gaps (FVG) as Targets
```
Bullish FVG: Gap between candle 1 high and candle 3 low (in uptrend)
  → Target: fill midpoint of gap or top of gap
  → Entry: retest of gap midpoint in uptrend
  → SL: below candle 1 low

Bearish FVG: Gap between candle 1 low and candle 3 high (in downtrend)
  → Target: fill midpoint of gap or bottom of gap
```

---

## 7. Combined Method: Precise Entry/SL/TP Framework

### Step-by-Step Calculation
```
1. IDENTIFY MARKET STRUCTURE (higher TF)
   → Determine trend direction (1H/4H)
   → Mark BOS, CHoCH, OB, FVG zones

2. CALCULATE PIVOTS (same TF as entry)
   → Standard pivots for R/S levels
   → Camarilla for scalping entries

3. FIND FIBONACCI ZONES
   → Retracement for optimal entry (0.5-0.618)
   → Extensions for TP targets (1.272, 1.618, 2.0)

4. MEASURE ATR FOR RISK
   → SL = entry ± ATR × multiplier
   → Position size = risk$ / (ATR × mult)

5. CHECK VOLUME PROFILE
   → Confirm entry near POC/VAL/VAH
   → Check for HVN support/resistance confluence

6. LOOK FOR AB=CD MEASURED MOVE
   → Add confluence if AB=CD aligns with Fib extension

7. SET TARGETS
   → TP1: nearest confluence of Fib + Pivot + Volume level
   → TP2: next confluence zone
   → TP3: 1.618+ extension or structural target
```

### Confluence Scoring System
```
For each potential target, score:
  +1  Fibonacci extension level present
  +1  Pivot point (R/S) nearby (within 0.3× ATR)
  +1  Volume profile level (POC/VAH/VAL)
  +1  Market structure level (OB/FVG/BOS)
  +1  Measured move equality (AB=CD)

Score 4-5: High confidence target
Score 3:   Medium confidence
Score 1-2: Weak target (partial profit zone)
```

### Risk:Reward Minimums
```
Score 5: Accept 2:1 R:R minimum
Score 4: Accept 2.5:1 R:R minimum
Score 3: Require 3:1 R:R minimum
Score 2: Require 4:1 R:R minimum (or skip)
Score 1: Skip trade

With BTC ATR=$177:
  SL = 1.5 × $177 = $266
  For 3:1 R:R → minimum TP = $798 (4.5× ATR)
  For 2:1 R:R → minimum TP = $532 (3× ATR)
```
