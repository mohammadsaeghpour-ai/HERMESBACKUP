# Risk Management & Position Sizing for Small Capital Crypto Trading

## Account Parameters

| Parameter | Value |
|-----------|-------|
| Account Capital | $10 |
| Max Leverage | 20x |
| Max Loss Per Trade | $1.50 (15% of capital) |
| Max Daily Loss | $3.00 (30% of capital) |
| Assets | BTC, ETH |
| BTC-ETH Correlation | 0.89 |

---

## 1. Kelly Criterion

### Formula
```
f* = (p × b − q) / b
```
Where:
- `f*` = fraction of bankroll to wager
- `p` = probability of winning
- `b` = win/loss ratio (avg win ÷ avg loss)
- `q` = 1 − p = probability of losing

### Practical Application (Half-Kelly)

Full Kelly is too aggressive for real trading. Use **Half-Kelly** (f*/2) or **Quarter-Kelly** (f*/4).

| Scenario | p | b | Full Kelly | Half Kelly | Quarter Kelly |
|----------|------|------|-----------|------------|---------------|
| Moderate edge, 2:1 R:R | 0.55 | 2.0 | 32.5% | 16.25% | 8.13% |
| No edge, 2:1 R:R | 0.50 | 2.0 | 25.0% | 12.50% | 6.25% |
| Good edge, 1.5:1 R:R | 0.60 | 1.5 | 33.3% | 16.67% | 8.33% |
| Low win rate, 3:1 R:R | 0.45 | 3.0 | 26.7% | 13.33% | 6.67% |

### Rules for $10 Account
- **Always use Quarter-Kelly** (1/4 Kelly) for small accounts. Kelly assumes large samples; small capital = high variance.
- Track your last 20+ trades to estimate p and b.
- If you don't have enough data, default to conservative: risk 3-5% per trade.
- Kelly percentage applies to **margin**, not notional value.

**Example**: Half-Kelly = 16.25%, your margin = $10 × 0.1625 = **$1.625**, notional = $1.625 × 20 = **$32.50**

---

## 2. Fixed Fractional Position Sizing

### Formula
```
Margin per trade = Account × Risk Fraction
Position Size (notional) = Margin × Leverage

Risk per trade = Margin × Stop Distance (in %)
```

### Risk Fraction Table

| Risk Fraction | Margin | Notional (20x) | Max Loss at 5% Stop |
|---------------|--------|-----------------|---------------------|
| 1% ($0.10) | $0.10 | $2.00 | $0.10 |
| 2% ($0.20) | $0.20 | $4.00 | $0.20 |
| 3% ($0.30) | $0.30 | $6.00 | $0.30 |
| 5% ($0.50) | $0.50 | $10.00 | $0.50 |
| 10% ($1.00) | $1.00 | $20.00 | $1.00 |
| 15% ($1.50) | $1.50 | $30.00 | $1.50 |

### Recommended Rule
```
FIXED RISK PER TRADE = $0.30 (3% of capital)
```
This gives you:
- ~33 losing trades before account death (at 3% per trade)
- 10 trades max per day before hitting $3 daily loss limit
- Enough margin ($0.30 × 20 = $6.00 notional) to trade with most exchanges' minimums

### Reverse Position Sizing (stop-based)
Instead of fixed fraction, size based on where your stop is:
```
Margin = Max Loss Per Trade / Stop Distance %
```
**Example**: If stop is 1% from entry:
- Margin = $1.50 / 0.01 = $150 → **exceeds account** → trade is too tight, skip it
**Example**: If stop is 7% from entry:
- Margin = $1.50 / 0.07 = $21.43 → **exceeds account** → can only use partial
- Use available margin: min($21.43, $10) = **$10** → risk = $10 × 0.07 = $0.70 per trade

---

## 3. Anti-Martingale (Position Sizing After Wins/Losses)

### Core Principle
**Increase** size after wins, **decrease** after losses. The opposite of Martingale.

### Formula
```
Adjusted Size = Base Size × (1 + k)^n
```
Where:
- `k` = scaling factor (0.25 to 0.50 for small accounts)
- `n` = consecutive wins (positive) or losses (negative)

### Scaling Rules

| Consecutive Result | Multiplier | Margin (Base = $0.30) |
|--------------------|-----------|------------------------|
| -3 losses | 0.42× | $0.13 |
| -2 losses | 0.56× | $0.17 |
| -1 loss | 0.75× | $0.23 |
| 0 (baseline) | 1.00× | $0.30 |
| +1 win | 1.25× | $0.38 |
| +2 wins | 1.56× | $0.47 |
| +3 wins | 1.95× | $0.59 |
| +4 wins (max) | 2.44× | $0.73 |

### Hard Caps (Non-Negotiable for Small Capital)
```
MAX POSITION AFTER WINS = $0.75 margin (25% of account)
MAX POSITION AFTER LOSSES = $0.13 margin (1.3% of account)
RESET after 5 consecutive results (win or loss streak broken)
NEVER increase size beyond base after max daily loss hit
```

### Anti-Martingale vs Fixed Fractional Decision
- **Use Fixed Fractional** if you have < 50 trade history (more predictable)
- **Use Anti-Martingale** if you have proven edge (p > 0.55 and b > 1.5)
- **Combine**: Use fixed fractional as base, anti-mar as ±25% adjustment

---

## 4. Maximum Drawdown Rules

### Drawdown Tiers

| Drawdown Level | Account Value | Action |
|----------------|---------------|--------|
| 0-10% ($0-$1) | $10.00-$9.00 | Normal trading, 3% risk per trade |
| 10-20% ($1-$2) | $9.00-$8.00 | Reduce to 2% risk per trade |
| 20-30% ($2-$3) | $8.00-$7.00 | Reduce to 1% risk per trade, max 5 trades/day |
| 30-40% ($3-$4) | $7.00-$6.00 | **STOP TRADING** for 24 hours minimum |
| 40-50% ($4-$5) | $6.00-$5.00 | **STOP TRADING** for 72 hours, journal review required |
| >50% ($5+) | <$5.00 | **STOP TRADING** for 1 week, strategy reassessment mandatory |

### Daily Drawdown Rules
```
MAX DAILY LOSS = $3.00 (30% of capital)

Hit $3 daily loss:
├── STOP all trading immediately
├── Do NOT trade for remainder of the day
├── Journal: what went wrong?
├── Next day: trade at 50% normal size
└── If 3 consecutive daily max-loss days → stop for 72 hours
```

### Recovery Rules
```
After drawdown reduction:
├── 20% DD → trade at 1% risk until back to 10% DD level
├── 30% DD → trade at 0.5% risk until back to 20% DD level
├── Recovery must be PROVEN (3+ profitable days) before scaling back up
└── Never skip recovery tiers — climb back gradually
```

### Position-Level Circuit Breakers
```
Stop after 3 consecutive losses in a row → pause 2 hours
Stop after hitting max daily loss → done for the day
Stop if a single position moves against you > 2× intended stop → review execution
```

---

## 5. Correlation-Based Position Sizing (BTC + ETH Together)

### The Problem
BTC and ETH correlation = 0.89. When one drops, the other almost certainly drops too. Holding both positions means you're essentially double-betting on the same direction.

### Portfolio Volatility Formula
```
σ_portfolio = √(w₁²σ₁² + w₂²σ₂² + 2w₁w₂ρσ₁σ₂)

Where:
  w₁, w₂ = weight of BTC, ETH in portfolio
  σ₁, σ₂ = daily volatility of BTC, ETH
  ρ = correlation coefficient (0.89)
```

### Typical Values
- σ_BTC ≈ 3.5% daily (14-period ATR as % of price)
- σ_ETH ≈ 4.5% daily
- ρ = 0.89

### Portfolio Volatility by Weight Split

| BTC Weight | ETH Weight | Portfolio σ | Relative to Single Asset |
|-----------|-----------|-------------|-------------------------|
| 100% | 0% | 3.50% | 1.00× |
| 70% | 30% | 3.67% | 1.05× |
| 50% | 50% | 3.89% | 1.11× |
| 30% | 70% | 4.19% | 1.20× |
| 0% | 100% | 4.50% | 1.29× |

### Correlation Scaling Factor
```
Correlation Multiplier = √(1 + 2ρ × w₁ × w₂ / (w₁² + w₂²))

For 50/50 split with ρ=0.89:
  Multiplier = √(1 + 2 × 0.89 × 0.25 / 0.50) = √(1.89) = 1.375

Effective position reduction = 1 / 1.375 = 0.727 (reduce each by 27%)
```

### Rules for BTC + ETH Together

```
RULE 1: NEVER exceed 100% of account in combined notional exposure
  i.e., BTC margin + ETH margin ≤ $10 total

RULE 2: When trading BOTH simultaneously:
  Total risk budget = Account × 3% = $0.30
  Split: BTC gets 55% ($0.165), ETH gets 45% ($0.135)
  (Give more to BTC = lower vol asset)

RULE 3: Correlation-Adjusted Position Size:
  Adjusted_margin = Base_margin / Correlation_Multiplier
  Example: Base $0.30 → Adjusted = $0.30 / 1.375 = $0.218

RULE 4: Maximum simultaneous positions = 2
  (BTC + ETH only; never add a third correlated crypto)

RULE 5: Same-direction positions (both long or both short):
  Reduce total margin by 40% (because correlation amplifies risk)
  → Total margin = $0.30 × 0.60 = $0.18 split across both

RULE 6: Opposite-direction positions (hedged):
  May use full margin since they partially offset
  → But this is NOT recommended at small capital (locking up margin)
```

### Practical Allocation
```
Preferred: Trade ONE asset at a time
  → Use full $0.30 margin on single best setup
  → Avoids correlation risk entirely

If must trade both:
  → BTC: $0.15 margin × 20x = $3.00 notional
  → ETH: $0.12 margin × 20x = $2.40 notional
  → Total margin: $0.27 (within $0.30 budget)
  → Combined max loss: ~$0.27 (if both hit stops)
```

---

## 6. Session-Based Risk Adjustment

### Session Definitions (UTC)

| Session | Hours (UTC) | Risk Multiplier | Notes |
|---------|------------|----------------|-------|
| Asian | 00:00-08:00 | **0.5×** | Low volume, thin books, stop hunts |
| London | 08:00-16:00 | **1.0×** | Standard, good liquidity |
| New York | 13:00-21:00 | **1.0×** | Standard, good liquidity |
| London-NY Overlap | 13:00-16:00 | **1.25×** | Best liquidity, tightest spreads |
| Late NY | 21:00-00:00 | **0.75×** | Winding down, liquidity drops |
| Weekend (Sat-Sun) | All day | **0.25×** | Extreme thinness, gaps, manipulation |
| Holiday | All day | **0.25×** | Low participation |

### Event-Based Adjustments

| Event | Risk Multiplier | Notes |
|-------|----------------|-------|
| FOMC announcement | **0.25×** (before) → **1.0×** (30min after) | Extreme vol before, trend after |
| NFP (Non-Farm Payroll) | **0.5×** | First 30min chaotic |
| CPI release | **0.5×** | First 30min chaotic |
| Crypto-specific (ETF, hack, exchange) | **0.0×** (pause) | Wait for clarity |
| High funding rate period | **0.75×** | Liquidation cascades likely |

### Effective Risk Per Session

Base risk per trade: **$0.30** (3% of account)

| Session | Multiplier | Effective Risk |
|---------|-----------|----------------|
| Asian | 0.5× | $0.15 |
| London/NY | 1.0× | $0.30 |
| Overlap | 1.25× | $0.375 |
| Late NY | 0.75× | $0.225 |
| Weekend | 0.25× | $0.075 |
| FOMC day (pre) | 0.25× | $0.075 |

### Decision Tree
```
Current time → Check session → Apply multiplier
    ↓
Event today? → Apply event multiplier (use LOWER of two)
    ↓
Check: is effective risk ≤ $1.50 max loss? → If yes, proceed
    ↓
Check: would this trade exceed $3 daily loss? → If yes, skip
    ↓
Enter trade with adjusted margin
```

---

## 7. Volatility-Adjusted Stops

### ATR-Based Stop Loss
```
Stop Distance = k × ATR(n)

Where:
  ATR = Average True Range (14-period recommended)
  k = multiplier (1.0 to 3.0 depending on strategy)
```

### Stop Distance → Position Size
```
Margin = Max Loss Per Trade / Stop Distance (as decimal)

Example:
  BTC price: $60,000
  ATR(14): $2,100 (3.5% of price)
  k = 2.0
  Stop distance: $4,200 (7.0% of price)
  
  Margin = $1.50 / 0.07 = $21.43 → exceeds account ($10)
  → Use max available margin: $10
  → Actual risk: $10 × 0.07 = $0.70 per trade ✓
```

### ATR Multiplier Guide

| Multiplier | BTC Stop % | ETH Stop % | Purpose | Margin Needed ($1.50 loss) |
|-----------|-----------|-----------|---------|---------------------------|
| k=1.0 | 3.5% | 4.5% | Scalping (very tight) | $42.86 / $33.33 |
| k=1.5 | 5.3% | 6.8% | Intraday | $28.57 / $22.22 |
| k=2.0 | 7.0% | 9.0% | Swing (recommended) | $21.43 / $16.67 |
| k=2.5 | 8.8% | 11.3% | Wide swing | $17.14 / $13.33 |
| k=3.0 | 10.5% | 13.5% | Position (not for small accts) | $14.29 / $11.11 |

### Small Account Reality Check
At $10 with 20x leverage:
- Your max notional per trade ≈ $6-30 (depending on risk fraction)
- A 7% stop means max loss = $0.42-$2.10
- **Best approach**: Use k=2.0 (ATR×2) stops, size to max loss of $0.30-$0.75

### Trailing Stop Formula
```
Trailing Stop = Entry ± (k × ATR)
Update stop every time price moves (k × ATR) in your favor

Example:
  Entry: $60,000 (long)
  Initial stop: $60,000 - (2 × $2,100) = $55,800
  Price moves to $64,200 (moved 2 ATRs)
  New stop: $64,200 - $4,200 = $60,000 (breakeven)
  Price moves to $68,400
  New stop: $68,400 - $4,200 = $64,200 (locked in profit)
```

### Volatility Regime Detection
```
Current ATR vs 50-period average ATR:

| Ratio | Regime | Action |
|-------|--------|--------|
| < 0.7 | Low vol / compression | Widen stops (k=2.5), reduce size |
| 0.7-1.3 | Normal vol | Standard stops (k=2.0), normal size |
| 1.3-2.0 | High vol | Tighten stops (k=1.5), reduce size by 50% |
| > 2.0 | Extreme vol / crisis | NO TRADE or k=1.0 with min size |
```

---

## Combined Position Sizing Formula

The master formula combines all factors:

```
FINAL MARGIN = Base_Risk × Kelly_Fraction × Session_Multiplier × Drawdown_Multiplier × Volatility_Multiplier

Where:
  Base_Risk = Account × 0.03 = $10 × 0.03 = $0.30
  Kelly_Fraction = min(Kelly_result / 2, 0.25)  [Quarter-Kelly cap]
  Session_Multiplier = 0.25 to 1.25 (from session table)
  Drawdown_Multiplier = 0.0 to 1.0 (from DD rules)
  Volatility_Multiplier = 0.5 to 1.0 (from vol regime)

Then:
  Stop_Distance = k × ATR(14)  [k=2.0 default]
  Max_Position = Final_Margin × Leverage
  Actual_Risk = min(Final_Margin, Max_Loss_Per_Trade / Stop_Distance)
```

### Example Walkthrough

```
Account: $10
Session: London-NY overlap (1.25×)
No drawdown yet (1.0×)
Normal volatility (1.0×)
55% win rate, 2:1 R:R → Quarter-Kelly = 8.13% → use 8%

Final_Margin = $0.30 × 0.08/0.03 × 1.25 × 1.0 × 1.0
             = $0.30 × 2.67 × 1.25
             = $1.00

Stop distance: ATR×2 = 7.0% (BTC)
Max risk check: $1.00 × 0.07 = $0.07 per trade ✓ (well under $1.50)
Notional: $1.00 × 20 = $20.00
```

---

## Quick Reference: Decision Flowchart

```
START TRADE DECISION
│
├─ Is daily loss < $3? ──── NO → STOP for day
│
├─ Is drawdown < 30%? ──── NO → Reduce to 0.5-1% risk or STOP
│
├─ What session? ────────── Apply multiplier
│
├─ Any events today? ────── Apply event multiplier (use lower)
│
├─ Current vol regime? ──── Apply vol multiplier
│
├─ Calculate Final Margin using master formula
│
├─ Correlation check:
│   ├─ Only 1 position open? → Use full margin
│   └─ 2 positions open? → Split margin (55/45), reduce by 40%
│
├─ Position size = min(Final_Margin × Leverage, available_margin)
│
├─ Stop = Entry ± (2 × ATR)
│
├─ Verify: Actual_risk ≤ $1.50? ── NO → Reduce size or skip
│
├─ Set stop loss BEFORE entry
│
├─ EXECUTE TRADE
│
└─ Post-trade: Update drawdown tracker, win/loss counter
```

---

## Key Rules Summary (Hardcoded, Non-Negotiable)

1. **Never risk more than $1.50 (15%) on a single trade**
2. **Never lose more than $3.00 (30%) in a single day**
3. **Stop at 30% total drawdown** — mandatory 24h pause
4. **Trade one asset at a time** when possible (avoids correlation risk)
5. **Always set stop loss** before entry (ATR×2 minimum)
6. **Reduce size by 27%** when holding correlated positions simultaneously
7. **Quarter-Kelly maximum** for position sizing calculations
8. **Session multipliers are mandatory** — don't trade full size in thin markets
9. **3 consecutive losses** → pause trading for 2 hours minimum
10. **Recovery from drawdown** is gradual — never jump back to full size

---

## Anti-Martingale Streak Tracker

Track wins and losses consecutively (reset when streak breaks):

```
Streak Count: [___]
Current Multiplier: [___]

After each trade:
  Win  → Streak + 1, multiplier = 1.25^(min(streak, 4))
  Loss → Streak - 1, multiplier = 0.75^(min(abs(streak), 3))
  Reset to streak=0 and multiplier=1.0 after 5 consecutive same results
```

---

## Daily Risk Budget Tracker

```
Date: ___________
Starting Capital: $__________

Trades:
  #1: [___] +/-$___ Running total: $___
  #2: [___] +/-$___ Running total: $___
  #3: [___] +/-$___ Running total: $___
  ...

Daily Loss Limit: $3.00
Daily P/L: $_______
Status: □ Under limit  □ Hit limit → STOP
```
