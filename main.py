import os
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
  .analysis {{ white-space: pre-wrap; line-height: 1.6; }}
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
    if "买入" in text: return "badge-buy"
    elif "卖出" in text: return "badge-sell"
    else: return "badge-hold"

# ---------- 主逻辑 ----------
md_lines = []
html_parts = []

for stock in stocks:
    # Tavily 搜索
    try:
        search_result = tavily_client.search(query=f"{stock} 股票 近期新闻 行情", max_results=5)
        news_items = [f"- {r['title']}: {r['content'][:200]}" for r in search_result.get("results", [])]
        news_text = "\n".join(news_items)
    except Exception as e:
        news_text = f"搜索新闻失败: {e}"
        news_items = [news_text]

    # DeepSeek 分析
    prompt = f"""你是一位专业的股票分析师。请根据下面的近期资讯，对股票 {stock} 做出简短分析。
资讯：
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

    # 解析评级
    rating = "持有"
    for line in analysis.split("\n"):
        if line.startswith("评级") or line.startswith("- 评级"):
            if "买入" in line: rating = "买入"
            elif "卖出" in line: rating = "卖出"
            break
    badge_class = get_badge_class(rating)

    # Markdown 附件
    md_lines.append(f"## {stock}\n**评级：** {rating}\n\n**资讯摘要：**\n{news_text}\n\n**AI 分析：**\n{analysis}\n\n---\n")

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

# 保存文件
md_content = f"# 股票分析报告\n\n生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n" + "\n".join(md_lines)
with open("report.md", "w", encoding="utf-8") as f: f.write(md_content)

html_content = html_header.format(time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + "\n".join(html_parts) + html_footer
with open("report.html", "w", encoding="utf-8") as f: f.write(html_content)

print("报告已生成：report.md 和 report.html")
