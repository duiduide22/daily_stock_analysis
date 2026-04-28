import os
from datetime import datetime, timedelta
from openai import OpenAI
from tavily import TavilyClient
import akshare as ak
import pandas as pd

# ---------- 初始化 ----------
deepseek_key = os.environ["DEEPSEEK_API_KEY"]
deepseek_base = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com")
tavily_key = os.environ["TAVILY_API_KEY"]

ds_client = OpenAI(api_key=deepseek_key, base_url=deepseek_base)
tavily_client = TavilyClient(api_key=tavily_key)

stock_input = os.environ.get("STOCK_LIST", "SH.600519")
stocks = [s.strip() for s in stock_input.split(",") if s.strip()]

# ---------- HTML 样式与页头 ----------
html_header = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif; padding: 20px; background-color: #f5f7fa; }}
  .report {{ max-width: 700px; margin: 0 auto; background: white; border-radius: 12px; padding: 30px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }}
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
  .risk {{ color: #d4380d; }}
  hr {{ border: none; border-top: 1px solid #f0f0f0; margin: 25px 0 10px; }}
</style>
</head>
<body>
<div class="report">
<h1>📈 股票分析报告</h1>
<p class="time">生成时间：{time}</p>
"""

html_footer = """</div></body></html>"""


# ---------- 评级徽章 ----------
def get_badge_class(text):
    if "买入" in text:
        return "badge-buy"
    elif "卖出" in text:
        return "badge-sell"
    else:
        return "badge-hold"


# ---------- 获取股票技术面与基本面数据 ----------
def get_stock_data(symbol):
    """
    从 akshare 获取技术面（近期K线）和基本面（PE/PB）数据。
    输入 symbol 格式: 'SH.600519' 或 'SZ.000858'
    返回: dict 包含 price_data, pe, pb, 或错误信息
    """
    try:
        # 解析市场与代码
        if "." in symbol:
            market, code = symbol.split(".")
            market = market.upper()
        else:
            # 默认按上海处理
            market, code = "SH", symbol

        # ----- 获取近期日K线（最近30个交易日）-----
        if market == "SH":
            full_code = f"sh{code}"
        else:
            full_code = f"sz{code}"

        # 使用 akshare 的 stock_zh_a_hist 接口
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=60)).strftime("%Y%m%d")  # 取60天以防停牌
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        if df.empty:
            return {"error": "未获取到行情数据"}

        # 取最近30条
        df = df.tail(30)
        # 简单计算：最新价、5日均价、10日均价、20日均价、近期涨跌幅
        latest = df.iloc[-1]
        close_series = df["收盘"].astype(float)
        ma5 = close_series.rolling(5).mean().iloc[-1]
        ma10 = close_series.rolling(10).mean().iloc[-1]
        ma20 = close_series.rolling(20).mean().iloc[-1]
        pct_change_5d = (close_series.iloc[-1] / close_series.iloc[-6] - 1) * 100 if len(close_series) >= 6 else None
        pct_change_20d = (close_series.iloc[-1] / close_series.iloc[-21] - 1) * 100 if len(close_series) >= 21 else None

        price_info = {
            "最新价": latest["收盘"],
            "5日均价": round(ma5, 2) if not pd.isna(ma5) else None,
            "10日均价": round(ma10, 2) if not pd.isna(ma10) else None,
            "20日均价": round(ma20, 2) if not pd.isna(ma20) else None,
            "5日涨跌幅(%)": round(pct_change_5d, 2) if pct_change_5d is not None else None,
            "20日涨跌幅(%)": round(pct_change_20d, 2) if pct_change_20d is not None else None,
            "最高价": latest["最高"],
            "最低价": latest["最低"],
            "成交量": latest["成交量"],
        }

        # ----- 获取市盈率、市净率（使用 stock_a_lg_indicator 接口）-----
        try:
            ind_df = ak.stock_a_lg_indicator(symbol=code)
            if not ind_df.empty:
                latest_ind = ind_df.iloc[-1]
                pe = latest_ind.get("pe", None)
                pb = latest_ind.get("pb", None)
            else:
                pe, pb = None, None
        except Exception:
            pe, pb = None, None

        return {
            "price_info": price_info,
            "PE": pe,
            "PB": pb,
        }

    except Exception as e:
        return {"error": f"获取数据失败: {e}"}


# ---------- 格式化数据为文本 ----------
def format_stock_data_text(data):
    if "error" in data:
        return f"数据获取失败：{data['error']}"
    text = "【技术面数据】\n"
    if "price_info" in data:
        pi = data["price_info"]
        for k, v in pi.items():
            text += f"{k}: {v}\n"
    text += "\n【基本面估值】\n"
    text += f"市盈率(PE): {data.get('PE', 'N/A')}\n"
    text += f"市净率(PB): {data.get('PB', 'N/A')}\n"
    return text


# ---------- 格式化数据为 HTML 表格 ----------
def format_stock_data_html(data):
    if "error" in data:
        return f'<p>数据获取失败：{data["error"]}</p>'
    html = '<table class="data-table"><tr><th>指标</th><th>数值</th></tr>'
    if "price_info" in data:
        pi = data["price_info"]
        for k, v in pi.items():
            html += f"<tr><td>{k}</td><td>{v}</td></tr>"
    html += f"<tr><td>市盈率(PE)</td><td>{data.get('PE', 'N/A')}</td></tr>"
    html += f"<tr><td>市净率(PB)</td><td>{data.get('PB', 'N/A')}</td></tr>"
    html += "</table>"
    return html


# ---------- 主分析逻辑 ----------
md_lines = []
html_parts = []

for stock in stocks:
    # ---- 获取股票数据 ----
    stock_data = get_stock_data(stock)
    data_text = format_stock_data_text(stock_data)
    data_html = format_stock_data_html(stock_data)

    # ---- Tavily 搜索 ----
    try:
        search_result = tavily_client.search(query=f"{stock} 股票 近期新闻 行情", max_results=5)
        news_items = [f"- {r['title']}: {r['content'][:200]}" for r in search_result.get("results", [])]
        news_text = "\n".join(news_items)
    except Exception as e:
        news_text = f"搜索新闻失败: {e}"
        news_items = [news_text]

    # ---- DeepSeek 分析（包含技术面和基本面）----
    prompt = f"""你是一位专业的股票分析师。请根据以下数据对该股票 {stock} 做出分析。

【市场数据】
{data_text}

【近期资讯】
{news_text}

请严格按三行输出：
评级：（买入/持有/卖出）
核心逻辑：一句话说明理由
风险提示：一句话说明主要风险"""
    try:
        response = ds_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        analysis = response.choices[0].message.content
    except Exception as e:
        analysis = f"分析失败: {e}"

    # ---- 解析评级 ----
    rating = "持有"
    for line in analysis.split("\n"):
        if line.startswith("评级") or line.startswith("- 评级"):
            if "买入" in line:
                rating = "买入"
            elif "卖出" in line:
                rating = "卖出"
            else:
                rating = "持有"
            break

    badge_class = get_badge_class(rating)

    # ---- 构建 Markdown 报告 ----
    md_lines.append(f"## {stock}\n")
    md_lines.append(f"**评级：** {rating}\n\n")
    md_lines.append(f"**市场数据：**\n{data_text}\n\n")
    md_lines.append(f"**资讯摘要：**\n{news_text}\n\n")
    md_lines.append(f"**AI 分析：**\n{analysis}\n\n---\n")

    # ---- 构建 HTML 卡片 ----
    html_parts.append(f"""
    <div class="stock-card">
      <div class="stock-title">{stock} <span class="badge {badge_class}">{rating}</span></div>
      <div class="section">
        <div class="section-title">📊 市场数据</div>
        {data_html}
      </div>
      <div class="section">
        <div class="section-title">📰 资讯摘要</div>
        <ul class="news-list">
          {''.join(f'<li>{item.lstrip("- ")}</li>' for item in news_items)}
        </ul>
      </div>
      <div class="section">
        <div class="section-title">🤖 AI 分析</div>
        <div class="analysis">{analysis}</div>
      </div>
    </div>
    """)

# ---------- 保存 Markdown 附件 ----------
md_content = f"# 股票分析报告\n\n生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n" + "\n".join(md_lines)
with open("report.md", "w", encoding="utf-8") as f:
    f.write(md_content)

# ---------- 保存 HTML 报告 ----------
html_content = html_header.format(time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + "\n".join(html_parts) + html_footer
with open("report.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("报告已生成：report.md 和 report.html")
