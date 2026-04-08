import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# --- 1. 設定網頁標題 ---
st.set_page_config(page_title="智能投資組合優化器", layout="wide")
st.title('📈 智能投資組合優化器 (融初效益版)')
st.markdown("""
此工具提供華爾街等級的投資組合分析，包含 **風險控管**、**融資模擬** 與 **複合基準指數對照**。
""")

# --- 2. 參數設定 ---
st.sidebar.header('參數設定')
tickers_input = st.sidebar.text_input('股票/基金代號 (請用空白隔開)', 'VFIAX VBTLX TSLA NVDA')
user_tickers = tickers_input.upper().split()

# 基準指數設定
st.sidebar.markdown("---")
st.sidebar.header("⚖️ 基準指數設定 (Benchmark)")
bench_input = st.sidebar.text_input(
    '基準代號與權重 (格式: 代號:%)', 
    'SPY:60 AGG:40', 
    help="請用冒號指定權重，並用空白隔開。\n例如：\n1. 股債平衡: SPY:60 AGG:40\n2. 純美股: SPY"
)

years = st.sidebar.slider('回測年數', 1, 20, 10)
risk_free_rate = 0.02 

# --- 融資設定 ---
st.sidebar.markdown("---")
st.sidebar.header("💰 融資設定 (Margin)")
use_margin = st.sidebar.checkbox("開啟融資回測模式")

if use_margin:
    loan_ratio = st.sidebar.slider("融資成數 (銀行借款比例)", 0.0, 0.9, 0.6, 0.1)
    margin_rate = st.sidebar.number_input("融資年利率 (%)", 2.0, 15.0, 6.0, 0.1) / 100
    self_fund_ratio = 1 - loan_ratio
    leverage = 1 / self_fund_ratio if self_fund_ratio > 0 else 1
    st.sidebar.info(f"槓桿倍數：**{leverage:.1f} 倍**")
else:
    loan_ratio = 0.0
    margin_rate = 0.0
    leverage = 1.0

# --- 3. 核心邏輯 ---
if st.sidebar.button('開始計算'):
    if len(user_tickers) < 2:
        st.error("請至少輸入兩檔標的。")
    else:
        with st.spinner('正在計算融資效益與回測數據...'):
            try:
                # ==========================
                # A. 數據準備
                # ==========================
                end_date = datetime.today()
                start_date = end_date - timedelta(days=365*years)
                
                # 1. 下載使用者投資組合
                data = yf.download(user_tickers, start=start_date, end=end_date, auto_adjust=True)
                
                if 'Close' in data.columns:
                    df_close = data['Close']
                else:
                    df_close = data
                
                df_close.dropna(inplace=True)
                
                if df_close.empty:
                    st.error("無法抓取投資組合數據。")
                    st.stop()
                
                tickers = df_close.columns.tolist()

                # 2. 下載與合成 Benchmark
                bench_config = []
                try:
                    items = bench_input.strip().split()
                    for item in items:
                        if ':' in item:
                            parts = item.split(':')
                            ticker = parts[0].upper()
                            weight = float(parts[1])
                        else:
                            ticker = item.upper()
                            weight = 100.0 
                        bench_config.append({'ticker': ticker, 'weight': weight})
                    
                    total_bench_w = sum([x['weight'] for x in bench_config])
                    if total_bench_w == 0: total_bench_w = 1
                    for x in bench_config:
                        x['weight'] /= total_bench_w
                    
                    bench_tickers = [x['ticker'] for x in bench_config]
                    bench_weights = [x['weight'] for x in bench_config]

                except Exception as e:
                    st.error(f"基準指數格式錯誤: {e}")
                    st.stop()

                bench_data_raw = yf.download(bench_tickers, start=start_date, end=end_date, auto_adjust=True)
                
                if 'Close' in bench_data_raw.columns:
                    df_bench_raw = bench_data_raw['Close']
                else:
                    df_bench_raw = bench_data_raw
                
                if isinstance(df_bench_raw, pd.Series):
                    df_bench_raw = df_bench_raw.to_frame(name=bench_tickers[0])

                # 日期對齊
                common_index = df_close.index.intersection(df_bench_raw.index)
                df_close = df_close.loc[common_index]
                df_bench_raw = df_bench_raw.loc[common_index]
                
                if df_bench_raw.empty:
                    normalized_bench = None
                else:
                    bench_daily_ret = df_bench_raw.pct_change().fillna(0)
                    try:
                        aligned_bench_ret = bench_daily_ret[bench_tickers]
                        composite_bench_ret = aligned_bench_ret.dot(bench_weights)
                    except:
                        composite_bench_ret = bench_daily_ret.mean(axis=1)

                    normalized_bench = (1 + composite_bench_ret).cumprod()
                    normalized_bench.name = "基準指數 (Benchmark)"

                # 統計數據
                returns = df_close.pct_change().dropna()
                cov_matrix = returns.cov() * 252
                mean_returns = returns.mean() * 252
                corr_matrix = returns.corr()
                
                num_assets = len(tickers)
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                bounds = tuple((0, 1) for _ in range(num_assets))
                init_guess = [1/num_assets] * num_assets
                
                normalized_prices = df_close / df_close.iloc[0]

                # 函數庫
                def calculate_mdd(series):
                    roll_max = series.cummax()
                    drawdown = (series - roll_max) / roll_max
                    return drawdown.min()

                def calculate_margin_equity(raw_portfolio_value, leverage, loan_ratio, annual_rate):
                    if leverage == 1:
                        return raw_portfolio_value
                    
                    debt = leverage - 1
                    daily_rate = annual_rate / 365 
                    position_value = raw_portfolio_value * leverage
                    interest_cost = pd.Series(np.arange(len(raw_portfolio_value)) * debt * daily_rate, index=raw_portfolio_value.index)
                    margin_equity = position_value - debt - interest_cost
                    return margin_equity

                st.success("運算完成！")

                # ==========================
                # B. 分頁顯示
                # ==========================
                tab1, tab2 = st.tabs(["🛡️ 最小風險組合", "🚀 最大夏普組合"])

                # --- 繪圖函數 ---
                def plot_performance(port_val, strategy_name, color):
                    bench_label = f"基準 ({bench_input})"
                    fig = px.line(port_val, title=f'資產成長回測')
                    fig.update_traces(line=dict(color=color, width=3), name=strategy_name)
                    
                    if normalized_bench is not None:
                        # 這是您遇到錯誤的地方，已經修正為 .ffill()
                        aligned_bench = normalized_bench.reindex(port_val.index).ffill()
                        if aligned_bench.iloc[0] > 0:
                            aligned_bench = aligned_bench / aligned_bench.iloc[0]
                        fig.add_trace(go.Scatter(x=aligned_bench.index, y=aligned_bench, 
                                                 mode='lines', name=bench_label, 
                                                 line=dict(color='gray', width=2, dash='dash')))
                    
                    st.plotly_chart(fig, use_container_width=True)

                # --- Tab 1: 最小風險 ---
                with tab1:
                    st.subheader("🛡️ 最小風險組合 (GMV)")
                    if use_margin:
                        st.caption(f"⚠️ **融資模式**：槓桿 {leverage:.1f} 倍 | 年利率 {margin_rate:.1%}")

                    def min_variance(weights, cov_matrix):
                        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    
                    res_min = minimize(min_variance, init_guess, args=(cov_matrix,), 
                                       method='SLSQP', bounds=bounds, constraints=constraints)
                    w_min = res_min.x
                    
                    exp_ret_min = np.sum(mean_returns * w_min)
                    exp_vol_min = res_min.fun
                    
                    col1_1, col1_2 = st.columns([1, 2])
                    with col1_1:
                        st.markdown("### 📊 預期績效")
                        c1, c2 = st.columns(2)
                        c1.metric("預期報酬", f"{exp_ret_min:.2%}")
                        c2.metric("預期波動", f"{exp_vol_min:.2%}")
                        st.divider()
                        
                        clean_w = [round(w, 4) if w > 0.0001 else 0.0 for w in w_min]
                        df_min = pd.DataFrame({'標的': tickers, '配置': clean_w})
                        df_min['顯示權重'] = df_min['配置'].apply(lambda x: f"{x:.1%}")
                        df_min = df_min.sort_values('配置', ascending=False)
                        st.table(df_min[['標的', '顯示權重']])
                        
                        fig_pie = px.pie(df_min[df_min['配置']>0], values='配置', names='標的', hole=0.4)
                        fig_pie.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with col1_2:
                        raw_port_val = (normalized_prices * w_min).sum(axis=1)
                        margin_port_val = calculate_margin_equity(raw_port_val, leverage, loan_ratio, margin_rate)
                        margin_port_val.name = "🛡️ 策略淨值"
                        
                        plot_performance(margin_port_val, "🛡️ 最小風險組合", "green")
                        
                        # ★ 計算融資效益差異 (關鍵修改)
                        total_ret = margin_port_val.iloc[-1] - 1
                        raw_total_ret = raw_port_val.iloc[-1] - 1 # 沒融資的報酬
                        
                        cagr = (margin_port_val.iloc[-1])**(1/years) - 1 if margin_port_val.iloc[-1] > 0 else -1
                        mdd = calculate_mdd(margin_port_val)
                        
                        # Benchmark 報酬
                        if normalized_bench is not None:
                            bench_total_ret = normalized_bench.iloc[-1]/normalized_bench.iloc[0] - 1
                        else:
                            bench_total_ret = 0
                        
                        # 決定顯示什麼 Delta
                        if use_margin:
                            # 如果開融資，顯示「比不融資多賺多少」
                            margin_diff = total_ret - raw_total_ret
                            delta_msg = f"融資效益: {margin_diff:+.2%}"
                            delta_color = "normal" # 綠漲紅跌
                        else:
                            # 沒融資，顯示「比大盤強多少」
                            delta_msg = f"vs Benchmark: {total_ret - bench_total_ret:+.2%}"
                            delta_color = "normal"

                        st.markdown("### 💰 回測結果")
                        cb1, cb2, cb3 = st.columns(3)
                        cb1.metric("總報酬率", f"{total_ret:.2%}", delta=delta_msg, delta_color=delta_color)
                        cb2.metric("年化報酬", f"{cagr:.2%}")
                        cb3.metric("最大回撤", f"{mdd:.2%}", delta="注意風險", delta_color="inverse")

                # --- Tab 2: 最大夏普 ---
                with tab2:
                    st.subheader("🚀 最大夏普組合 (Max Sharpe)")
                    if use_margin:
                        st.caption(f"⚠️ **融資模式**：槓桿 {leverage:.1f} 倍 | 年利率 {margin_rate:.1%}")

                    def neg_sharpe_ratio(weights, mean_returns, cov_matrix, rf):
                        p_ret = np.sum(mean_returns * weights)
                        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                        return - (p_ret - rf) / p_vol
                    
                    args = (mean_returns, cov_matrix, risk_free_rate)
                    res_sharpe = minimize(neg_sharpe_ratio, init_guess, args=args,
                                          method='SLSQP', bounds=bounds, constraints=constraints)
                    w_sharpe = res_sharpe.x
                    
                    exp_ret_sharpe = np.sum(mean_returns * w_sharpe)
                    exp_vol_sharpe = np.sqrt(np.dot(w_sharpe.T, np.dot(cov_matrix, w_sharpe)))
                    sharpe_ratio = (exp_ret_sharpe - risk_free_rate) / exp_vol_sharpe

                    col2_1, col2_2 = st.columns([1, 2])
                    with col2_1:
                        st.markdown("### 📊 預期績效")
                        c_s1, c_s2 = st.columns(2)
                        c_s1.metric("預期報酬", f"{exp_ret_sharpe:.2%}")
                        c_s2.metric("預期波動", f"{exp_vol_sharpe:.2%}")
                        st.metric("夏普值", f"{sharpe_ratio:.2f}", delta="優異")
                        st.divider()

                        clean_w_s = [round(w, 4) if w > 0.0001 else 0.0 for w in w_sharpe]
                        df_sharpe = pd.DataFrame({'標的': tickers, '配置': clean_w_s})
                        df_sharpe['顯示權重'] = df_sharpe['配置'].apply(lambda x: f"{x:.1%}")
                        df_sharpe = df_sharpe.sort_values('配置', ascending=False)
                        st.table(df_sharpe[['標的', '顯示權重']])
                        
                        fig_pie_s = px.pie(df_sharpe[df_sharpe['配置']>0], values='配置', names='標的', hole=0.4)
                        fig_pie_s.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
                        st.plotly_chart(fig_pie_s, use_container_width=True)

                    with col2_2:
                        raw_port_val_s = (normalized_prices * w_sharpe).sum(axis=1)
                        margin_port_val_s = calculate_margin_equity(raw_port_val_s, leverage, loan_ratio, margin_rate)
                        margin_port_val_s.name = "🚀 策略淨值"
                        
                        plot_performance(margin_port_val_s, "🚀 最大夏普組合", "red")
                        
                        # ★ 計算融資效益差異
                        total_ret_s = margin_port_val_s.iloc[-1] - 1
                        raw_total_ret_s = raw_port_val_s.iloc[-1] - 1
                        
                        cagr_s = (margin_port_val_s.iloc[-1])**(1/years) - 1 if margin_port_val_s.iloc[-1] > 0 else -1
                        mdd_s = calculate_mdd(margin_port_val_s)
                        
                        if normalized_bench is not None:
                            bench_total_ret = normalized_bench.iloc[-1]/normalized_bench.iloc[0] - 1
                        else:
                            bench_total_ret = 0
                            
                        # 決定顯示什麼 Delta
                        if use_margin:
                            margin_diff_s = total_ret_s - raw_total_ret_s
                            delta_msg_s = f"融資效益: {margin_diff_s:+.2%}"
                        else:
                            delta_msg_s = f"vs Benchmark: {total_ret_s - bench_total_ret:+.2%}"

                        st.markdown("### 💰 回測結果")
                        csb1, csb2, csb3 = st.columns(3)
                        csb1.metric("總報酬率", f"{total_ret_s:.2%}", delta=delta_msg_s, delta_color="normal")
                        csb2.metric("年化報酬", f"{cagr_s:.2%}")
                        csb3.metric("最大回撤", f"{mdd_s:.2%}", delta="注意風險", delta_color="inverse")

            except Exception as e:
                st.error(f"發生錯誤：{str(e)}")
else:
    st.info("請在左側輸入股票代號並按下「開始計算」")

st.sidebar.markdown("---")
st.sidebar.caption("⚠️ **免責聲明**")
st.sidebar.caption("""
本工具僅供市場分析與模擬參考，不構成任何投資建議或邀約。
融資交易涉及高風險，可能導致損失超過原始本金。
歷史績效不代表未來獲利保證。
""")
