import os
import sys
from datetime import datetime
from openai import OpenAI
from tavily import TavilyClient

# 从环境变量读取密钥（由 GitHub Secrets 注入）
deepseek_key = os.environ["DEEPSEEK_API_KEY"]
deepseek_base = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com")
tavily_key = os.environ["TAVILY_API_KEY"]

# 初始化客户端
ds_client = OpenAI(api_key=deepseek_key, base_url=deepseek_base)
tavily_client = TavilyClient(api_key=tavily_key)

# 获取要分析的股票列表（从 Actions 输入变量）
stock_input = os.environ.get("STOCK_LIST", "SH.600519")
stocks = [s.strip() for s in stock_input.split(",") if s.strip()]

report_lines = []
report_lines.append(f"# 股票分析报告\n\n生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

for stock in stocks:
    report_lines.append(f"## {stock}\n")
    
    # 1. 使用 Tavily 搜索近期新闻
    try:
        search_result = tavily_client.search(query=f"{stock} 股票 近期新闻 行情", max_results=5)
        news_text = "\n".join([f"- {r['title']}: {r['content'][:200]}" for r in search_result.get("results", [])])
    except Exception as e:
        news_text = f"搜索新闻失败: {e}"
    
    # 2. 使用 DeepSeek 生成分析决策
    prompt = f"""
你是一位专业的股票分析师。请根据以下搜索到的近期资讯，对股票 {stock} 做出简短分析。
资讯：
{news_text}

请输出：
- 评级（买入/持有/卖出）
- 一句话核心逻辑
- 风险提示
回复请直接给出内容，不要解释。
"""
    try:
        response = ds_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        analysis = response.choices[0].message.content
    except Exception as e:
        analysis = f"分析失败: {e}"
    
    report_lines.append(f"**资讯摘要：**\n{news_text}\n\n**AI 分析：**\n{analysis}\n\n---\n")

# 保存报告文件
report_content = "\n".join(report_lines)
with open("report.md", "w", encoding="utf-8") as f:
    f.write(report_content)

print("分析完成，报告已保存为 report.md")
print(report_content)  # 同时在日志中输出，便于查看
