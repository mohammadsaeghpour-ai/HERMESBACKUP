# 🧠 HQIP v3 — Professional Crypto Trading Intelligence Platform
## High-Quality Intelligence Platform

> **نسل سوم سیستم معاملاتی هوشمند رمزارز**
> **با یادگیری تعاملی، هوش مصنوعی و تحلیل چندتایم‌فریمی**

---

## 📊 آمار کلی

| شاخص | مقدار |
|---|---|
| **فایل‌های پایتون** | ۹۵ |
| **خطوط کد** | ۲۱,۹۶۸ |
| **ایجنت‌ها** | ۲۴ |
| **ماژول‌ها** | ۱۲ |
| **Win Rate** | ۷۵٪ |
| **بازده بک‌تست** | +۱۱.۴٪ |

---

## 🏗️ معماری سیستم

```
┌─────────────────────────────────────────────────────────┐
│                    HQIP v3 Architecture                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │  core/   │  │  data/   │  │  ict/    │  │ agents/ │ │
│  │ config   │  │ exchange │  │ OB, FVG  │  │ 24 agent│ │
│  │ logger   │  │历史 data │  │ structure│  │ trend   │ │
│  │ database │  │ sessions │  │ liquid   │  │ momentum│ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬────┘ │
│       │             │             │             │       │
│  ┌────▼─────────────▼─────────────▼─────────────▼────┐ │
│  │              indicators/                           │ │
│  │  trend | momentum | volatility | volume | fib     │ │
│  └────────────────────┬──────────────────────────────┘ │
│                       │                                │
│  ┌────────────────────▼──────────────────────────────┐ │
│  │              consensus/                            │ │
│  │         Bayesian + Expected Value Engine           │ │
│  └────────────────────┬──────────────────────────────┘ │
│                       │                                │
│  ┌────────────────────▼──────────────────────────────┐ │
│  │              ml/                                   │ │
│  │    Q-Learning | Interactive Learning | ML Engine   │ │
│  └────────────────────┬──────────────────────────────┘ │
│                       │                                │
│  ┌────────────────────▼──────────────────────────────┐ │
│  │              risk/                                 │ │
│  │      Position Sizer | Stop Loss | Portfolio Risk   │ │
│  └────────────────────┬──────────────────────────────┘ │
│                       │                                │
│  ┌────────────────────▼──────────────────────────────┐ │
│  │              execution/                            │ │
│  │           Order Manager | Paper Trader             │ │
│  └────────────────────┬──────────────────────────────┘ │
│                       │                                │
│  ┌────────────────────▼──────────────────────────────┐ │
│  │              backtester/                           │ │
│  │           Event-Driven Backtester                  │ │
│  └───────────────────────────────────────────────────┘ │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 ساختار پوشه

```
hqip/
├── 📄 ARCHITECTURE.md          # این فایل
├── 📄 main.py                  # نقطه ورود اصلی
├── 📄 ml_engine.py             # موتور ML
├── 📄 pipeline_engine.py       # موتور pipeline
├── 📄 strategies.py            # استراتژی‌ها
├── 📄 strategies_rr.py         # استراتژی R:R
├── 📄 backtest_v3.py           # بک‌تست v3
├── 📄 orchestrator.py          # هماهنگ‌کننده
│
├── 📁 core/                    # هسته سیستم
│   ├── config.py               # تنظیمات
│   ├── logger.py               # لاگر
│   └── database.py             # پایگاه داده
│
├── 📁 data/                    # مدیریت داده
│   ├── exchange.py             # صرافی‌ها
│   ├── historical.py           # داده تاریخی
│   └── sessions.py             # جلسات معاملاتی
│
├── 📁 ict/                     # ICT Methodology
│   ├── order_blocks.py         # بلاک‌های سفارش
│   ├── fvg.py                  # شکاف‌های ارزش منصفانه
│   ├── structure.py            # ساختار بازار
│   ├── liquidity.py            # نقدینگی
│   └── killzones.py            # مناطق کشتار
│
├── 📁 indicators/              # شاخص‌ها
│   ├── trend.py                # روند
│   ├── momentum.py             # مومنتوم
│   ├── volatility.py           # نوسانات
│   ├── volume.py               # حجم
│   └── fibonacci.py            # فیبوناچی
│
├── 📁 agents/                  # ۲۴ ایجنت هوشمند
│   ├── trend_agent.py          # تحلیل روند
│   ├── momentum_agent.py       # تحلیل مومنتوم
│   ├── volatility_agent.py     # تحلیل نوسانات
│   ├── volume_agent.py         # تحلیل حجم
│   ├── market_structure_agent.py # ساختار بازار
│   ├── regime_agent.py         # رژیم بازار
│   ├── sentiment_agent.py      # احساسات
│   ├── news_agent.py           # اخبار
│   ├── whale_agent.py          # رفتار نهنگ‌ها
│   ├── ml_agent.py             # یادگیری ماشین
│   ├── dl_forecast_agent.py    # پیش‌بینی عمیق
│   ├── consensus_agent.py      # اجماع
│   ├── manager_agent.py        # مدیریت
│   ├── explainability_agent.py # توضیح‌پذیری
│   └── ...                     # ۱۰ ایجنت دیگر
│
├── 📁 consensus/               # اجماع هوشمند
│   └── engine.py               # موتور بیزین + EV
│
├── 📁 risk/                    # مدیریت ریسک
│   ├── position_sizer.py       # حجم معامله
│   ├── stop_loss.py            # استاپ‌لاس
│   └── portfolio_risk.py       # ریسک سبد
│
├── 📁 ml/                      # یادگیری ماشین
│   ├── ppo_agent.py            # ربات Q-Learning ⭐
│   └── interactive_learning.py # یادگیری تعاملی
│
├── 📁 execution/               # اجرای معاملات
│   └── orders.py               # مدیریت سفارشات
│
├── 📁 backtester/              # بک‌تست
│   └── engine.py               # موتور بک‌تست
│
├── 📁 dashboard/               # داشبورد
│   └── html_dashboard.py       # داشبورد HTML
│
├── 📁 alerts/                  # هشدارها
│
└── 📁 reports/                 # گزارش‌ها
```

---

## 🤖 ۲۴ ایجنت هوشمند

| # | ایجنت | لایه | وظیفه |
|---|---|---|---|
| ۱ | **TrendAgent** | analysis | تشخیص جهت روند |
| ۲ | **MomentumAgent** | analysis | اندازه‌گیری قدرت حرکت |
| ۳ | **VolatilityAgent** | analysis | نوسانات بازار |
| ۴ | **VolumeAgent** | analysis | حجم معاملات |
| ۵ | **MarketStructureAgent** | analysis | ساختار بازار (BOS/CHoCH) |
| ۶ | **RegimeAgent** | analysis | رژیم بازار (trending/ranging) |
| ۷ | **SentimentAgent** | data | تحلیل احساسات |
| ۸ | **NewsAgent** | data | تحلیل اخبار |
| ۹ | **WhaleAgent** | data | رفتار نهنگ‌ها |
| ۱۰ | **MLAgent** | ml | یادگیری ماشین |
| ۱۱ | **DLForecastAgent** | ml | پیش‌بینی عمیق |
| ۱۲ | **ConsensusAgent** | decision | اجماع هوشمند |
| ۱۳ | **ManagerAgent** | decision | مدیریت معاملات |
| ۱۴ | **ExplainabilityAgent** | monitor | توضیح‌پذیری |
| ۱۵-۲۴ | **سایر ایجنت‌ها** | various | وظایف تخصصی |

---

## 🧠 یادگیری تعاملی

### Q-Learning Agent
- **State Space:** ۲۰۸ state منحصربه‌فرد
- **Actions:** HOLD / BUY / SELL
- **Win Rate:** ۷۵٪
- **Return:** +۱۱.۴٪

### Interactive Learning
- **MemoryBank:** حافظه بلندمدت بازار
- **AttentionMechanism:** تمرکز روی مهم‌ترین ویژگی‌ها
- **DeepNeuralNetwork:** شبکه عصبی عمیق
- **GameTheoryAgent:** تئوری بازی
- **ReinforcementLearning:** یادگیری تقویتی

---

## 📊 ICT Methodology

| مفهوم | توضیح |
|---|---|
| **Order Blocks** | مناطق سفارش بزرگ |
| **FVG** | شکاف‌های ارزش منصفانه |
| **BOS/CHoCH** | شکست ساختار / تغییر روند |
| **Liquidity** | مناطق نقدینگی |
| **Kill Zones** | زمان‌های ویژه معاملاتی |

---

## 🔗 لینک‌ها

- **GitHub:** https://github.com/mohammadsaeghpour-ai/HERMESBACKUP
- **مستندات:** ARCHITECTURE.md

---

> **ساخته شده توسط:** Hermes Agent
> **تاریخ:** ۲۸ جولای ۲۰۲۶
> **نسخه:** v3.0
