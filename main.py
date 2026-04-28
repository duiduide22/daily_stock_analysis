import os
import re
from datetime import datetime, timedelta
from openai import OpenAI
from tavily import TavilyClient
import baostock as bs
import pandas as pd

# ---------- 初始化 ----------
deepseek_key = os.environ["DEEPSEEK_API_KEY"]
deepseek_base = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com")
tavily_key = os.environ["TAVILY_API_KEY"]

ds_client = OpenAI(api_key=deepseek_key, base_url=deepseek_base)
tavily_client = TavilyClient(api_key=tavily_key)

stock_input = os.environ.get("STOCK_LIST", "SH.600519")
stocks = [s.strip() for s in stock_input.split(",") if s.strip()]

# ---------- HTML 样式 ----------
html_header = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif; padding: 20px; background-color: #f5f7fa; }}
  .report {{ max-width: 750px; margin: 0 auto; background: white; border-radius: 12px; padding: 30px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }}
  h1 {{ color: #1a1a1a; font-size: 24px; margin-top: 0; }}
  .time {{ color: #8c8c8c; font-size: 14px; margin-bottom: 30px; }}
  .stock-card {{ border: 1px solid #e8e8e8; border-radius: 10px; padding: 20px; margin-bottom: 25px; background: #fafafa; }}
  .stock-title {{ font-size: 20px; font-weight: 600; margin-bottom: 10px; color: #262626; }}
  .badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 14px; font-weight: 600; color: white; }}
  .badge-buy {{ background-color: #4caf50; }}
  .badge-hold {{ background-color: #ff9800; }}
  .badge-sell {{ background-color: #f44336; }}
  .section {{ margin-top: 15px; }}
  .section-title {{ font-weight: 600; color: #595959; margin-bottom: 6px; }}
  .news-list {{ list-style: none; padding: 0; margin: 0; }}
  .news-list li {{ margin-bottom: 8px; border-left: 3px solid #1890ff; padding-left: 10px; color: #434343; font-size: 14px; }}
  .data-table {{ width: 100%; border-collapse: collapse; margin-bottom: 10px; font-size: 14px; }}
  .data-table td, .data-table th {{ border: 1px solid #e8e8e8; padding: 6px 10px; }}
  .data-table th {{ background: #f0f0f0; }}
  .analysis {{ white-space: pre-wrap; line-height: 1.6; }}
  .compare-card {{ background: #eef6ff; border-left: 5px solid #1890ff; padding: 15px 20px; border-radius: 8px; margin-top: 30px; }}
  hr {{ border: none; border-top: 1px solid #f0f0f0; margin: 25px 0 10px; }}
</style>
</head>
<body>
<div class="report">
<h1>📈 股票分析报告</h1>
<p class="time">生成时间：{time}</p>
"""
html_footer = """</div></body></html>"""


def get_badge_class(text):
    if "买入" in text:
        return "badge-buy"
    elif "卖出" in text:
        return "badge-sell"
    else:
        return "badge-hold"


def clean_news(items, min_len=20):
    """清洗新闻列表：去除乱码、过短内容"""
    cleaned = []
    for item in items:
        text = item.lstrip("- ").strip()
        if len(text) < min_len:
            continue
        if len(re.findall(r'[\u4e00-\u9fffA-Za-z0-9]', text)) < len(text) * 0.4:
            continue
        cleaned.append(item)
    return cleaned


def parse_rating(analysis):
    rating = "持有"
    for line in analysis.split("\n"):
        if "评级" in line:
            if "买入" in line:
                rating = "买入"
            elif "卖出" in line:
                rating = "卖出"
            else:
                rating = "持有"
            break
    return rating


# ========== 技术面数据获取（BaoStock）==========
def get_technical_data(symbol):
    """
    使用 BaoStock 获取 A 股日K线，并计算常用技术指标。
    symbol 格式: 'SH.600519' 或 'SZ.002881'
    返回: 技术指标字典，或含 error 字段的字典
    """
    try:
        bs.login()
        market, code = symbol.split(".")
        market_lower = market.lower()

        # 获取近一年日K线（后复权，以便计算真实涨跌幅）
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        rs = bs.query_history_k_data_plus(
            f"{market_lower}.{code}",
            "date,open,high,low,close,volume,amount",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2"   # 2=后复权，方便计算涨跌幅
        )
        df = rs.get_data()
        bs.logout()

        if df.empty or len(df) < 20:
            return {"error": "K线数据不足，无法计算技术指标"}

        # 数据类型转换
        df = df.astype({"close": float, "high": float, "low": float, "volume": float})
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values
        dates = df["date"].values

        latest = df.iloc[-1]
        start_price = close[0]

        # ---- 均线 ----
        def sma(arr, n):
            if len(arr) < n:
                return None
            return round(pd.Series(arr).rolling(n).mean().iloc[-1], 2)

        ma5 = sma(close, 5)
        ma10 = sma(close, 10)
        ma20 = sma(close, 20)
        ma60 = sma(close, 60)

        # ---- MACD (12,26,9) ----
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_bar = 2 * (dif - dea)
        macd_val = round(macd_bar.iloc[-1], 3)
        macd_signal = "金叉" if macd_bar.iloc[-1] > 0 and macd_bar.iloc[-2] <= 0 else ("死叉" if macd_bar.iloc[-1] < 0 and macd_bar.iloc[-2] >= 0 else "持续")

        # ---- RSI (14) ----
        delta = pd.Series(close).diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        avg_gain = up.rolling(14).mean().iloc[-1]
        avg_loss = down.rolling(14).mean().iloc[-1]
        rsi = 100.0 if avg_loss == 0 else round(100 - 100 / (1 + avg_gain / avg_loss), 1)

        # ---- KDJ (9,3,3) ----
        low_n = pd.Series(low).rolling(9).min()
        high_n = pd.Series(high).rolling(9).max()
        rsv = (close - low_n) / (high_n - low_n) * 100
        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        j = 3 * k - 2 * d
        k_val = round(k.iloc[-1], 1)
        d_val = round(d.iloc[-1], 1)
        j_val = round(j.iloc[-1], 1)

        # ---- 波动率 ----
        returns = pd.Series(close).pct_change().dropna()
        volatility = round(returns.std() * (252 ** 0.5) * 100, 2) if len(returns) > 0 else None

        # ---- 涨跌幅 ----
        change_1d = round((close[-1] / close[-2] - 1) * 100, 2) if len(close) >= 2 else None
        change_5d = round((close[-1] / close[-6] - 1) * 100, 2) if len(close) >= 6 else None
        change_20d = round((close[-1] / close[-21] - 1) * 100, 2) if len(close) >= 21 else None
        total_change = round((close[-1] / close[0] - 1) * 100, 2)

        return {
            "最新价": latest["close"],
            "MA5": ma5,
            "MA10": ma10,
            "MA20": ma20,
            "MA60": ma60,
            "MACD": macd_val,
            "MACD信号": macd_signal,
            "RSI": rsi,
            "K": k_val, "D": d_val, "J": j_val,
            "年化波动率(%)": volatility,
            "1日涨跌幅(%)": change_1d,
            "5日涨跌幅(%)": change_5d,
            "20日涨跌幅(%)": change_20d,
            "区间涨跌幅(%)": total_change,
            "数据区间": f"{dates[0]} 至 {dates[-1]}"
        }
    except Exception as e:
        try:
            bs.logout()
        except:
            pass
        return {"error": f"技术指标计算失败: {e}"}


# ---------- 格式化技术指标为文本/HTML ----------
def format_tech_text(data):
    if "error" in data:
        return f"技术面数据获取失败：{data['error']}"
    lines = ["【技术面数据】"]
    for k, v in data.items():
        lines.append(f"   {k}: {v}")
    return "\n".join(lines)


def format_tech_html(data):
    if "error" in data:
        return f'<p style="color:#d4380d;">技术面数据获取失败：{data["error"]}</p>'
    html = '<table class="data-table"><tr><th>指标</th><th>数值</th></tr>'
    for k, v in data.items():
        html += f"<tr><td>{k}</td><td>{v}</td></tr>"
    html += "</table>"
    return html


# ---------- 横向对比 ----------
def compare_stocks(stock_results):
    if len(stock_results) < 2:
        return ""
    summary = "\n".join([f"{s}: {r} (RSI={t.get('RSI','N/A')})" for s, r, t, _ in stock_results])
    prompt = f"""以下是几只股票的分析结论与技术指标摘要，请横向比较并给出投资优先级建议。

{summary}

请输出：
1. 综合排序（从优到劣）
2. 配置建议（哪只适合进攻，哪只适合防守）
3. 整体策略（一句话）"""
    try:
        response = ds_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"对比分析失败: {e}"


# ========== 单股分析 ==========
def analyze_single(stock):
    # ---- 技术面 ----
    tech_data = get_technical_data(stock)
    tech_text = format_tech_text(tech_data)

    # ---- 消息面（Tavily）----
    try:
        search_result = tavily_client.search(query=f"{stock} 股票 近期新闻 行情", max_results=5)
        raw_items = [f"- {r['title']}: {r['content'][:200]}" for r in search_result.get("results", [])]
        news_items = clean_news(raw_items)
        if not news_items:
            news_items = ["- 暂无有效资讯"]
        news_text = "\n".join(news_items)
    except Exception as e:
        news_text = f"搜索新闻失败: {e}"
        news_items = [news_text]

    # ---- DeepSeek AI 综合分析（技术面 + 消息面）----
    prompt = f"""你是一位资深股票分析师。请结合技术面指标和近期资讯，对股票 {stock} 做出分析。

【技术面数据】
{tech_text}

【近期资讯】
{news_text}

请严格按照以下格式输出，每个字段一行：
评级：（买入/持有/卖出）
核心逻辑：（1-2句话，说明主要驱动或压制因素，须引用技术指标或新闻）
市场情绪：（积极/中性/谨慎）
短期展望：（未来1-4周可能走势）
风险提示：（1-2个主要风险，含具体原因）"""
    try:
        response = ds_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        analysis = response.choices[0].message.content
    except Exception as e:
        analysis = f"分析失败: {e}"

    rating = parse_rating(analysis)
    return stock, rating, tech_data, news_items, analysis


# ========== 主流程 ==========
stock_results = []
md_lines = []
html_parts = []

for stock in stocks:
    stock, rating, tech_data, news_items, analysis = analyze_single(stock)
    stock_results.append((stock, rating, tech_data, analysis))
    badge_class = get_badge_class(rating)
    tech_html = format_tech_html(tech_data)

    # Markdown
    md_lines.append(f"## {stock}\n**评级：** {rating}\n\n**技术指标：**\n{format_tech_text(tech_data)}\n\n**资讯摘要：**\n" + "\n".join(news_items) + f"\n\n**AI 分析：**\n{analysis}\n\n---\n")

    # HTML
    html_parts.append(f"""
    <div class="stock-card">
      <div class="stock-title">{stock} <span class="badge {badge_class}">{rating}</span></div>
      <div class="section">
        <div class="section-title">📊 技术面指标</div>
        {tech_html}
      </div>
      <div class="section">
        <div class="section-title">📰 资讯摘要</div>
        <ul class="news-list">
          {''.join(f'<li>{item.lstrip("- ")}</li>' for item in news_items)}
        </ul>
      </div>
      <div class="section">
        <div class="section-title">🤖 AI 综合分析</div>
        <div class="analysis">{analysis}</div>
      </div>
    </div>
    """)

# 横向对比
if len(stocks) >= 2:
    compare_text = compare_stocks(stock_results)
    md_lines.append(f"\n# 横向对比总览\n\n{compare_text}")
    html_parts.append(f"""
    <div class="compare-card">
      <div class="section-title" style="font-size:16px;">🔍 横向对比总览</div>
      <div class="analysis">{compare_text}</div>
    </div>
    """)

# ---------- 保存文件 ----------
md_content = f"# 股票分析报告\n\n生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n" + "\n".join(md_lines)
with open("report.md", "w", encoding="utf-8") as f:
    f.write(md_content)

html_content = html_header.format(time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + "\n".join(html_parts) + html_footer
with open("report.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("报告已生成：report.md 和 report.html")
