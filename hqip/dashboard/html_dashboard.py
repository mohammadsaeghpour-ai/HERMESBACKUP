"""
HQIP HTML Dashboard — Standalone trading terminal
"""
from datetime import datetime

def generate_dashboard(signals: list, trades: list, metrics: dict, equity_curve: list) -> str:
    """Generate standalone HTML dashboard."""
    
    # Build signal rows
    signal_rows = ""
    for s in signals[:20]:
        direction = s.get('direction', '?')
        color = '#16a34a' if direction == 'BUY' else '#e71d36' if direction == 'SELL' else '#6b7280'
        d_fa = '🟢 خرید' if direction == 'BUY' else '🔴 فروش' if direction == 'SELL' else '⚪ خنثی'
        signal_rows += f"""<tr>
            <td>{s.get('symbol','')}</td>
            <td>{s.get('tf','')}</td>
            <td style="color:{color};font-weight:bold">{d_fa}</td>
            <td>{s.get('confidence',0):.0f}%</td>
            <td>${s.get('entry',0):,.2f}</td>
            <td>${s.get('sl',0):,.2f}</td>
            <td>${s.get('tp1',0):,.2f}</td>
            <td>${s.get('tp2',0):,.2f}</td>
            <td>${s.get('tp3',0):,.2f}</td>
        </tr>"""
    
    # Build trade rows
    trade_rows = ""
    for t in trades[-20:]:
        pnl = t.get('pnl', 0)
        color = '#16a34a' if pnl > 0 else '#e71d36'
        trade_rows += f"""<tr>
            <td>{t.get('symbol','')}</td>
            <td>{'🟢' if t.get('direction')=='BUY' else '🔴'}</td>
            <td>${t.get('entry_price',0):,.2f}</td>
            <td>${t.get('exit_price',0):,.2f}</td>
            <td style="color:{color};font-weight:bold">${pnl:+.2f}</td>
            <td>{t.get('reason','')}</td>
        </tr>"""
    
    # Equity chart (simple SVG)
    equity_svg = _build_equity_chart(equity_curve)
    
    # Metrics
    m = metrics
    
    return f"""<!DOCTYPE html>
<html dir="rtl" lang="fa">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="30">
<title>HQIP v3 — داشبورد معاملاتی</title>
<style>
body {{ background:#0a0a0a; color:#e5e5e5; font-family:monospace; margin:0; padding:20px; }}
h1 {{ color:#22c55e; text-align:center; border-bottom:2px solid #22c55e; padding-bottom:10px; }}
h2 {{ color:#3b82f6; margin-top:30px; }}
table {{ width:100%; border-collapse:collapse; margin:10px 0; }}
th {{ background:#1a1a2e; color:#22c55e; padding:8px; text-align:right; border-bottom:2px solid #333; }}
td {{ padding:6px 8px; border-bottom:1px solid #222; text-align:right; }}
tr:hover {{ background:#1a1a2e; }}
.metric {{ display:inline-block; background:#1a1a2e; padding:15px 25px; margin:5px; border-radius:8px; text-align:center; }}
.metric .value {{ font-size:24px; color:#22c55e; font-weight:bold; }}
.metric .label {{ font-size:12px; color:#888; }}
.chart {{ background:#111; padding:20px; border-radius:8px; margin:10px 0; }}
.footer {{ text-align:center; color:#555; margin-top:30px; padding:20px; }}
</style>
</head>
<body>
<h1>🎯 HQIP v3 — پلتفرم هوش مصنوعی معاملاتی</h1>
<p style="text-align:center;color:#888">آخرین بروزرسانی: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

<div style="text-align:center">
    <div class="metric"><div class="value">${m.get('final_capital', m.get('current_capital', 0)):,.2f}</div><div class="label">سرمایه فعلی</div></div>
    <div class="metric"><div class="value">{m.get('total_return_pct', 0):+.1f}%</div><div class="label">بازده کل</div></div>
    <div class="metric"><div class="value">{m.get('total_trades', 0)}</div><div class="label">تعداد معاملات</div></div>
    <div class="metric"><div class="value">{m.get('win_rate', 0):.0f}%</div><div class="label">نرخ برد</div></div>
    <div class="metric"><div class="value">{m.get('sharpe_ratio', 0):.2f}</div><div class="label">نسبت شارپ</div></div>
    <div class="metric"><div class="value">{m.get('max_drawdown_pct', 0):.1f}%</div><div class="label">حداکثر افت</div></div>
</div>

<div class="chart"><h2>📈 نمودار سرمایه</h2>{equity_svg}</div>

<h2>📡 سیگنال‌های اخیر</h2>
<table>
<tr><th>نماد</th><th>تایم‌فریم</th><th>جهت</th><th>اطمینان</th><th>ورود</th><th>استاپ</th><th>تارگت۱</th><th>تارگت۲</th><th>تارگت۳</th></tr>
{signal_rows}
</table>

<h2>📊 تاریخچه معاملات</h2>
<table>
<tr><th>نماد</th><th>جهت</th><th>ورود</th><th>خروج</th><th>سود/ضرر</th><th>دلیل</th></tr>
{trade_rows}
</table>

<div class="footer">HQIP v3 — سیستم توصیه‌گر معاملاتی مبتنی بر هوش مصنوعی | ⚠️ این سیستم مشاوره مالی نیست</div>
</body>
</html>"""


def _build_equity_chart(equity_curve: list) -> str:
    """Build simple SVG equity chart."""
    if not equity_curve or len(equity_curve) < 2:
        return "<p>داده‌ای موجود نیست</p>"
    
    eq = equity_curve[-200:]  # Last 200 points
    width, height = 800, 200
    min_eq = min(eq) * 0.99
    max_eq = max(eq) * 1.01
    
    if max_eq == min_eq:
        max_eq = min_eq + 1
    
    points = []
    for i, v in enumerate(eq):
        x = i / (len(eq) - 1) * width
        y = height - (v - min_eq) / (max_eq - min_eq) * height
        points.append(f"{x:.1f},{y:.1f}")
    
    color = '#22c55e' if eq[-1] >= eq[0] else '#e71d36'
    
    return f"""<svg width="100%" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
        <rect width="{width}" height="{height}" fill="#111"/>
        <polyline points="{' '.join(points)}" fill="none" stroke="{color}" stroke-width="2"/>
        <text x="10" y="20" fill="#888" font-size="12">${eq[0]:,.0f}</text>
        <text x="10" y="{height-5}" fill="#888" font-size="12">${eq[-1]:,.0f}</text>
    </svg>"""
