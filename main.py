import os
import re
import time
from datetime import datetime, timedelta
from openai import OpenAI
from tavily import TavilyClient
import baostock as bs
import akshare as ak
import pandas as pd
import numpy as np

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
    if "买入" in text: return "badge-buy"
    elif "卖出" in text: return "badge-sell"
    else: return "badge-hold"

def clean_news(items, min_len=20):
    cleaned = []
    for item in items:
        text = item.lstrip("- ").strip()
        if len(text) < min_len: continue
        if len(re.findall(r'[\u4e00-\u9fffA-Za-z0-9]', text)) < len(text) * 0.4: continue
        cleaned.append(item)
    return cleaned

def parse_rating(analysis):
    rating = "持有"
    for line in analysis.split("\n"):
        if "评级" in line:
            if "买入" in line: rating = "买入"
            elif "卖出" in line: rating = "卖出"
            break
    return rating

def parse_symbol(symbol):
    """安全解析市场与代码，仅按第一个 . 拆分"""
    symbol = symbol.strip()
    if "." in symbol:
        parts = symbol.split(".", 1)
        market = parts[0].upper()
        code = parts[1]
    else:
        market, code = "SH", symbol
    return market, code

# ==================== 缠论分析模块 ====================
def merge_k(df):
    df = df.reset_index(drop=True)
    new_rows = []
    i = 0
    while i < len(df):
        if i == 0:
            new_rows.append(df.iloc[i].to_dict())
            i += 1
            continue
        prev = new_rows[-1]
        curr = df.iloc[i].to_dict()
        if i >= 2:
            direction = 1 if new_rows[-1]["high"] > new_rows[-2]["high"] else -1
        else:
            direction = 1 if curr["high"] > prev["high"] else -1
        if (curr["high"] >= prev["high"] and curr["low"] <= prev["low"]) or \
           (curr["high"] <= prev["high"] and curr["low"] >= prev["low"]):
            if direction == 1:
                prev["high"] = max(prev["high"], curr["high"])
                prev["low"] = max(prev["low"], curr["low"])
            else:
                prev["high"] = min(prev["high"], curr["high"])
                prev["low"] = min(prev["low"], curr["low"])
            prev["date"] = curr["date"]
        else:
            new_rows.append(curr)
        i += 1
    return pd.DataFrame(new_rows)

def find_fx(df_merged):
    fxs = []
    for i in range(1, len(df_merged)-1):
        pre, cur, nxt = df_merged.iloc[i-1], df_merged.iloc[i], df_merged.iloc[i+1]
        if cur["high"] > pre["high"] and cur["high"] > nxt["high"]:
            fxs.append((i, 1))
        elif cur["low"] < pre["low"] and cur["low"] < nxt["low"]:
            fxs.append((i, -1))
    return fxs

def build_bi(fxs, df_merged):
    bi_list = []
    if len(fxs) < 2:
        return bi_list
    i = 0
    while i < len(fxs)-1:
        a, b = fxs[i], fxs[i+1]
        if a[1] != b[1] and abs(df_merged.iloc[b[0]]["high"] - df_merged.iloc[a[0]]["low"]) > 1e-6:
            bi_list.append((a[0], b[0], 1 if b[1]==1 else -1))
            i += 1
        else:
            i += 1
    filtered = []
    for bi in bi_list:
        if not filtered or filtered[-1][2] != bi[2]:
            filtered.append(bi)
    return filtered

def find_zhongshu(bi_list, df_merged):
    if len(bi_list) < 3:
        return None
    # 修复：从 len(bi_list)-3 开始，保证 i+2 不越界
    for i in range(len(bi_list) - 3, -1, -1):
        b1, b2, b3 = bi_list[i], bi_list[i+1], bi_list[i+2]
        if b1[2] != b2[2] and b2[2] != b3[2]:
            high_pool = [
                df_merged.iloc[b1[0]]["high"], df_merged.iloc[b2[0]]["high"],
                df_merged.iloc[b1[1]]["high"], df_merged.iloc[b2[1]]["high"],
                df_merged.iloc[b3[0]]["high"], df_merged.iloc[b3[1]]["high"]
            ]
            low_pool = [
                df_merged.iloc[b1[0]]["low"], df_merged.iloc[b2[0]]["low"],
                df_merged.iloc[b1[1]]["low"], df_merged.iloc[b2[1]]["low"],
                df_merged.iloc[b3[0]]["low"], df_merged.iloc[b3[1]]["low"]
            ]
            zg, zd = min(high_pool), max(low_pool)
            if zg >= zd:
                return (zd, zg)
    return None

def chanlun_analysis(df_raw):
    df = df_raw[["date","open","high","low","close"]].copy()
    df_merged = merge_k(df)
    fxs = find_fx(df_merged)
    bi_list = build_bi(fxs, df_merged)
    current_bi_dir = "无明确笔"
    if bi_list:
        current_bi_dir = "向上笔" if bi_list[-1][2] == 1 else "向下笔"
    zs = find_zhongshu(bi_list, df_merged)
    divergence = "无背驰"
    if zs and len(bi_list) >= 2:
        close_series = pd.Series(df_raw["close"].values)
        ema12 = close_series.ewm(span=12, adjust=False).mean()
        ema26 = close_series.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        enter_bi = bi_list[-2]
        leave_bi = bi_list[-1]
        s1, e1 = enter_bi[0], enter_bi[1]
        s2, e2 = leave_bi[0], leave_bi[1]
        area1 = abs(dif.iloc[s1:e1+1].sum())
        area2 = abs(dif.iloc[s2:e2+1].sum())
        if leave_bi[2] == 1 and df_raw.iloc[e2]["close"] > zs[1] and area2 < area1 * 0.9:
            divergence = "顶背驰风险"
        elif leave_bi[2] == -1 and df_raw.iloc[e2]["close"] < zs[0] and area2 < area1 * 0.9:
            divergence = "底背驰机会"
    return {
        "当前笔方向": current_bi_dir,
        "最近中枢": f"{zs[0]:.2f}-{zs[1]:.2f}" if zs else "无",
        "背驰状态": divergence
    }

# ==================== 基本面数据获取 ====================
def get_fundamental_data(symbol):
    try:
        bs.login()
        market, code = parse_symbol(symbol)
        market_lower = market.lower()

        # 成长性
        growth_rs = bs.query_growth_data(code=f"{market_lower}.{code}", year=2026, quarter=1)
        df_growth = growth_rs.get_data()
        if df_growth.empty:
            growth_rs = bs.query_growth_data(code=f"{market_lower}.{code}", year=2025, quarter=4)
            df_growth = growth_rs.get_data()
        if df_growth.empty:
            growth_rs = bs.query_growth_data(code=f"{market_lower}.{code}", year=2025, quarter=3)
            df_growth = growth_rs.get_data()

        # 盈利能力
        profit_rs = bs.query_profit_data(code=f"{market_lower}.{code}", year=2026, quarter=1)
        df_profit = profit_rs.get_data()
        if df_profit.empty:
            profit_rs = bs.query_profit_data(code=f"{market_lower}.{code}", year=2025, quarter=4)
            df_profit = profit_rs.get_data()
        if df_profit.empty:
            profit_rs = bs.query_profit_data(code=f"{market_lower}.{code}", year=2025, quarter=3)
            df_profit = profit_rs.get_data()

        # 估值
        valuation_rs = bs.query_history_k_data_plus(
            f"{market_lower}.{code}",
            "date,close,peTTM,pbMRQ",
            start_date=(datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
            end_date=datetime.now().strftime("%Y-%m-%d"),
            frequency="d",
            adjustflag="2"
        )
        df_val = valuation_rs.get_data()
        bs.logout()

        result = {}
        if not df_growth.empty:
            latest_g = df_growth.iloc[-1]
            result["营收同比增长(%)"] = latest_g.get("YOYOperateIncome", "N/A")
            result["净利润同比增长(%)"] = latest_g.get("YOYNetProfit", "N/A")
        else:
            result["营收同比增长(%)"] = "N/A"
            result["净利润同比增长(%)"] = "N/A"

        if not df_profit.empty:
            latest_p = df_profit.iloc[-1]
            result["ROE(%)"] = latest_p.get("ROE", "N/A")
        else:
            result["ROE(%)"] = "N/A"

        if not df_val.empty and len(df_val) > 0:
            latest_val = df_val.iloc[-1]
            result["市盈率TTM"] = latest_val.get("peTTM", "N/A")
            result["市净率"] = latest_val.get("pbMRQ", "N/A")
        else:
            result["市盈率TTM"] = "N/A"
            result["市净率"] = "N/A"

        return result
    except Exception as e:
        try: bs.logout()
        except: pass
        return {"error": str(e)}

# ==================== 技术面数据获取（BaoStock + akshare 备用）====================
def get_technical_data(symbol):
    """
    优先使用 BaoStock，如果数据不足或失败，自动切换到 akshare 获取数据。
    """
    def process_df(df):
        """通用的数据加工函数"""
        if df.empty or len(df) < 20:
            return {"error": "K线数据不足", "df_raw": None}

        df = df.astype({"close": float, "high": float, "low": float, "volume": float})
        close = df["close"].values; high = df["high"].values; low = df["low"].values; dates = df["date"].values

        def sma(arr, n):
            if len(arr) < n: return None
            return round(pd.Series(arr).rolling(n).mean().iloc[-1], 2)

        ma5 = sma(close, 5); ma10 = sma(close, 10); ma20 = sma(close, 20); ma60 = sma(close, 60)

        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_bar = 2 * (dif - dea)
        macd_val = round(macd_bar.iloc[-1], 3)
        macd_signal = "金叉" if macd_bar.iloc[-1] > 0 and macd_bar.iloc[-2] <= 0 else ("死叉" if macd_bar.iloc[-1] < 0 and macd_bar.iloc[-2] >= 0 else "持续")

        delta = pd.Series(close).diff()
        up = delta.clip(lower=0); down = -delta.clip(upper=0)
        avg_gain = up.rolling(14).mean().iloc[-1]
        avg_loss = down.rolling(14).mean().iloc[-1]
        rsi = 100 if avg_loss == 0 else round(100 - 100 / (1 + avg_gain / avg_loss), 1)

        low_n = pd.Series(low).rolling(9).min(); high_n = pd.Series(high).rolling(9).max()
        rsv = (close - low_n) / (high_n - low_n) * 100
        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        j = 3 * k - 2 * d
        k_val = round(k.iloc[-1], 1); d_val = round(d.iloc[-1], 1); j_val = round(j.iloc[-1], 1)

        returns = pd.Series(close).pct_change().dropna()
        volatility = round(returns.std() * (252 ** 0.5) * 100, 2) if len(returns) > 0 else None

        change_1d = round((close[-1] / close[-2] - 1) * 100, 2) if len(close) >= 2 else None
        change_5d = round((close[-1] / close[-6] - 1) * 100, 2) if len(close) >= 6 else None
        change_20d = round((close[-1] / close[-21] - 1) * 100, 2) if len(close) >= 21 else None
        total_change = round((close[-1] / close[0] - 1) * 100, 2)

        tech_dict = {
            "最新价": df.iloc[-1]["close"],
            "MA5": ma5, "MA10": ma10, "MA20": ma20, "MA60": ma60,
            "MACD": macd_val, "MACD信号": macd_signal,
            "RSI": rsi,
            "K": k_val, "D": d_val, "J": j_val,
            "年化波动率(%)": volatility,
            "1日涨跌幅(%)": change_1d,
            "5日涨跌幅(%)": change_5d,
            "20日涨跌幅(%)": change_20d,
            "区间涨跌幅(%)": total_change,
            "数据区间": f"{dates[0]} 至 {dates[-1]}",
            "df_raw": df
        }
        return tech_dict

    market, code = parse_symbol(symbol)

    # ----- 1. 尝试 BaoStock -----
    try:
        bs.login()
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        market_lower = market.lower()
        rs = bs.query_history_k_data_plus(
            f"{market_lower}.{code}",
            "date,open,high,low,close,volume",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2"
        )
        df = rs.get_data()
        bs.logout()

        if df.empty or len(df) < 20:
            # 缩短时间再试
            start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
            bs.login()
            rs2 = bs.query_history_k_data_plus(
                f"{market_lower}.{code}",
                "date,open,high,low,close,volume",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2"
            )
            df = rs2.get_data()
            bs.logout()

        if not df.empty and len(df) >= 20:
            return process_df(df)
    except Exception:
        try: bs.logout()
        except: pass

    # ----- 2. 切换到 akshare 备用 -----
    try:
        for attempt in range(2):
            try:
                df_ak = ak.stock_zh_a_hist(
                    symbol=code,
                    period="daily",
                    start_date=(datetime.now() - timedelta(days=365)).strftime("%Y%m%d"),
                    end_date=datetime.now().strftime("%Y%m%d"),
                    adjust="qfq"
                )
                if df_ak.empty:
                    continue
                # 统一列名
                df_ak = df_ak.rename(columns={
                    "日期": "date", "开盘": "open", "最高": "high",
                    "最低": "low", "收盘": "close", "成交量": "volume"
                })[["date", "open", "high", "low", "close", "volume"]]

                if len(df_ak) >= 20:
                    return process_df(df_ak)
            except Exception:
                time.sleep(2)
                continue
    except Exception:
        pass

    return {"error": "BaoStock与akshare均无法获取足够K线数据，该股票可能交易不活跃或数据源未覆盖", "df_raw": None}

# ==================== 格式化函数 ====================
def format_fundamental_text(data):
    if "error" in data:
        return f"基本面数据获取失败：{data['error']}"
    lines = ["【基本面数据】"]
    for k, v in data.items():
        lines.append(f"   {k}: {v}")
    return "\n".join(lines)

def format_fundamental_html(data):
    if "error" in data:
        return f'<p style="color:#d4380d;">基本面数据获取失败：{data["error"]}</p>'
    html = '<table class="data-table"><tr><th>指标</th><th>数值</th></tr>'
    for k, v in data.items():
        html += f"<tr><td>{k}</td><td>{v}</td></tr>"
    html += "</table>"
    return html

def format_tech_text(data):
    if "error" in data:
        return f"技术面数据获取失败：{data['error']}"
    lines = ["【技术面数据】"]
    for k, v in data.items():
        if k == "df_raw": continue
        lines.append(f"   {k}: {v}")
    return "\n".join(lines)

def format_tech_html(data):
    if "error" in data:
        return f'<p style="color:#d4380d;">技术面数据获取失败：{data["error"]}</p>'
    html = '<table class="data-table"><tr><th>指标</th><th>数值</th></tr>'
    for k, v in data.items():
        if k == "df_raw": continue
        html += f"<tr><td>{k}</td><td>{v}</td></tr>"
    html += "</table>"
    return html

def format_chanlun_text(data):
    if "error" in data:
        return f"缠论分析失败：{data['error']}"
    lines = ["【缠论结构】"]
    for k, v in data.items():
        lines.append(f"   {k}: {v}")
    return "\n".join(lines)

def format_chanlun_html(data):
    if "error" in data:
        return f'<p style="color:#d4380d;">缠论分析失败：{data["error"]}</p>'
    html = '<table class="data-table"><tr><th>指标</th><th>数值</th></tr>'
    for k, v in data.items():
        html += f"<tr><td>{k}</td><td>{v}</td></tr>"
    html += "</table>"
    return html

# ---------- 横向对比 ----------
def compare_stocks(stock_results):
    if len(stock_results) < 2:
        return ""
    summary = "\n".join([f"{s}: {r}" for s, r, _, _, _, _ in stock_results])
    prompt = f"""以下是几只股票的分析结论摘要，请横向比较并给出投资优先级建议。
{summary}
请输出：
1. 综合排序
2. 配置建议
3. 整体策略"""
    try:
        response = ds_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"对比分析失败: {e}"

# ========== 单股综合分析（智能降级）==========
def analyze_single(stock):
    fund_data = get_fundamental_data(stock)
    tech_result = get_technical_data(stock)
    if "error" in tech_result:
        tech_data = tech_result
        df_raw = None
    else:
        df_raw = tech_result.pop("df_raw")
        tech_data = tech_result

    if df_raw is not None:
        chanlun_data = chanlun_analysis(df_raw)
    else:
        chanlun_data = {"error": "缺少K线数据，无法进行缠论分析"}

    # 消息面
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

    # 智能降级 prompt：根据数据可用性自动调整
    has_fund = "error" not in fund_data
    has_tech = "error" not in tech_data
    has_chanlun = "error" not in chanlun_data

    prompt = f"你是一位资深股票分析师。请基于以下信息对股票 {stock} 做出分析。\n\n"
    if has_fund:
        prompt += f"【基本面数据】\n{format_fundamental_text(fund_data)}\n\n"
    else:
        prompt += "【基本面数据】暂未获取到，请基于技术和消息面判断。\n\n"
    if has_tech:
        prompt += f"【技术面数据】\n{format_tech_text(tech_data)}\n\n"
    else:
        prompt += "【技术面数据】暂未获取到，请基于基本面和消息面判断。\n\n"
    if has_chanlun:
        prompt += f"【缠论结构】\n{format_chanlun_text(chanlun_data)}\n\n"
    prompt += f"【近期资讯】\n{news_text}\n\n"
    prompt += """请严格按照以下格式输出：
评级：（买入/持有/卖出）
核心逻辑：（1-2句话）
市场情绪：（积极/中性/谨慎）
短期展望：（未来1-4周走势预判）
风险提示：（1-2个主要风险）"""

    try:
        response = ds_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        analysis = response.choices[0].message.content
    except Exception as e:
        analysis = f"分析失败: {e}"

    rating = parse_rating(analysis)
    return stock, rating, fund_data, tech_data, chanlun_data, news_items, analysis

# ---------- 主流程 ----------
stock_results = []
md_lines = []
html_parts = []

for stock in stocks:
    stock, rating, fund_data, tech_data, chanlun_data, news_items, analysis = analyze_single(stock)
    stock_results.append((stock, rating, fund_data, tech_data, chanlun_data, analysis))
    badge_class = get_badge_class(rating)

    # Markdown
    md_lines.append(f"## {stock}\n**评级：** {rating}\n\n"
                    f"**基本面：**\n{format_fundamental_text(fund_data)}\n\n"
                    f"**技术面：**\n{format_tech_text(tech_data)}\n\n"
                    f"**缠论结构：**\n{format_chanlun_text(chanlun_data)}\n\n"
                    f"**资讯摘要：**\n" + "\n".join(news_items) + f"\n\n**AI 分析：**\n{analysis}\n\n---\n")

    # HTML 卡片
    html_parts.append(f"""
    <div class="stock-card">
      <div class="stock-title">{stock} <span class="badge {badge_class}">{rating}</span></div>
      <div class="section">
        <div class="section-title">📊 基本面</div>
        {format_fundamental_html(fund_data)}
      </div>
      <div class="section">
        <div class="section-title">📈 技术面</div>
        {format_tech_html(tech_data)}
      </div>
      <div class="section">
        <div class="section-title">📐 缠论结构</div>
        {format_chanlun_html(chanlun_data)}
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

# ---------- 保存报告 ----------
md_content = f"# 股票分析报告\n\n生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n" + "\n".join(md_lines)
with open("report.md", "w", encoding="utf-8") as f:
    f.write(md_content)

html_content = html_header.format(time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + "\n".join(html_parts) + html_footer
with open("report.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("报告已生成：report.md 和 report.html")
