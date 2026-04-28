import os
import re
from datetime import datetime
from openai import OpenAI
from tavily import TavilyClient

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
    """清洗新闻列表：去除乱码、过短、纯数字字符等无效内容"""
    cleaned = []
    for item in items:
        text = item.lstrip("- ").strip()
        if len(text) < min_len:
            continue
        # 剔除含过多非中英文/数字符号的内容（简单判断）
        if len(re.findall(r'[\u4e00-\u9fffA-Za-z0-9]', text)) < len(text) * 0.4:
            continue
        cleaned.append(item)
    return cleaned

def parse_rating(analysis):
    """从分析文本中提取评级"""
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

# ---------- 单股分析函数 ----------
def analyze_single(stock):
    # 搜索
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

    # AI 分析（升级版 prompt）
    prompt = f"""你是一位资深股票分析师。请基于以下资讯对股票 {stock} 进行深入分析，并按指定格式输出。

【近期资讯】
{news_text}

请严格按照以下格式输出，每个字段用一行，不要有任何额外开场白：
评级：（买入/持有/卖出）
核心逻辑：（1-2句话，说明主要驱动因素或压制因素）
关键价位：（根据资讯中出现的价格数据，给出支撑位和压力位，若无提及则写“暂无数据”）
市场情绪：（积极/中性/谨慎）
短期展望：（未来1-4周可能走势）
风险提示：（1-2个主要风险）"""
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
    return stock, rating, news_items, analysis

# ---------- 横向对比函数 ----------
def compare_stocks(stock_results):
    if len(stock_results) < 2:
        return ""
    summary = "\n".join([f"{s}: {r}" for s, r, _, _ in stock_results])
    prompt = f"""以下是几只股票的分析结论，请横向比较并给出投资优先级建议。

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

# ---------- 主流程 ----------
stock_results = []
md_lines = []
html_parts = []

for stock in stocks:
    stock, rating, news_items, analysis = analyze_single(stock)
    stock_results.append((stock, rating, news_items, analysis))
    badge_class = get_badge_class(rating)

    # Markdown
    md_lines.append(f"## {stock}\n**评级：** {rating}\n\n**资讯摘要：**\n" + "\n".join(news_items) + f"\n\n**AI 分析：**\n{analysis}\n\n---\n")

    # HTML 卡片
    html_parts.append(f"""
    <div class="stock-card">
      <div class="stock-title">{stock} <span class="badge {badge_class}">{rating}</span></div>
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

# 横向对比（若有至少2只股票）
compare_text = ""
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
