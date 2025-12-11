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
st.title('📈 智能投資組合優化器 (旗艦銷售版)')
st.markdown("""
此工具提供華爾街等級的投資組合分析，專注於 **風險控管** 與 **資產配置效率**。
""")

# --- 2. 參數設定 ---
st.sidebar.header('參數設定')
tickers_input = st.sidebar.text_input('股票/基金代號 (請用空白隔開)', 'VFIAX VBTLX TSLA NVDA')
user_tickers = tickers_input.upper().split()
years = st.sidebar.slider('回測年數', 1, 20, 10)
risk_free_rate = 0.02 

# --- 3. 核心邏輯 ---
if st.sidebar.button('開始計算'):
    if len(user_tickers) < 2:
        st.error("請至少輸入兩檔標的。")
    else:
        with st.spinner('正在進行蒙地卡羅模擬與優化運算...'):
            try:
                # ==========================
                # A. 數據準備
                # ==========================
                end_date = datetime.today()
                start_date = end_date - timedelta(days=365*years)
                
                data = yf.download(user_tickers, start=start_date, end=end_date, auto_adjust=True)
                
                if 'Close' in data.columns:
                    df_close = data['Close']
                else:
                    df_close = data
                
                df_close.dropna(inplace=True)
                
                if df_close.empty:
                    st.error("無法抓取數據，請檢查代號。")
                    st.stop()

                tickers = df_close.columns.tolist()
                
                returns = df_close.pct_change().dropna()
                cov_matrix = returns.cov() * 252
                mean_returns = returns.mean() * 252
                corr_matrix = returns.corr()
                
                num_assets = len(tickers)
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                bounds = tuple((0, 1) for _ in range(num_assets))
                init_guess = [1/num_assets] * num_assets
                normalized_prices = df_close / df_close.iloc[0]

                # --------------------------
                # ★ 新增功能：MDD 計算函數
                # --------------------------
                def calculate_mdd(series):
                    # 計算累計最高點
                    roll_max = series.cummax()
                    # 計算當前價格與最高點的落差 (Drawdown)
                    drawdown = (series - roll_max) / roll_max
                    # 找出最大的落差 (MDD)
                    mdd = drawdown.min()
                    return mdd

                # --------------------------
                # ★ 新增功能：蒙地卡羅模擬 (效率前緣)
                # --------------------------
                num_portfolios = 2000 # 模擬 2000 種隨機組合
                results = np.zeros((3, num_portfolios)) # [報酬, 風險, 夏普]
                
                for i in range(num_portfolios):
                    w = np.random.random(num_assets)
                    w /= np.sum(w) # 歸一化
                    
                    p_ret = np.sum(mean_returns * w)
                    p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
                    
                    results[0,i] = p_ret
                    results[1,i] = p_vol
                    results[2,i] = (p_ret - risk_free_rate) / p_vol
                
                # 轉成 DataFrame 方便畫圖
                df_mc = pd.DataFrame(results.T, columns=['報酬率', '波動度', '夏普值'])

                st.success("運算完成！")

                # ==========================
                # B. 分頁顯示
                # ==========================
                tab1, tab2 = st.tabs(["🛡️ 最小風險組合 (保守)", "🚀 最大夏普值組合 (積極)"])

                # --- Tab 1: 最小風險 ---
                with tab1:
                    st.subheader("🛡️ 策略目標：極致抗跌")
                    
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
                        c1.metric("預期年化報酬", f"{exp_ret_min:.2%}")
                        c2.metric("預期年化波動", f"{exp_vol_min:.2%}", delta="極低", delta_color="normal")
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
                        port_val = (normalized_prices * w_min).sum(axis=1)
                        port_val.name = "🛡️ 最小風險組合"
                        combined = normalized_prices.copy()
                        combined["🛡️ 最小風險組合"] = port_val
                        
                        fig_line = px.line(combined, title=f'資產成長回測 (過去 {years} 年)')
                        fig_line.update_traces(line=dict(width=1), opacity=0.3)
                        fig_line.update_traces(selector=dict(name="🛡️ 最小風險組合"), line=dict(color='green', width=4), opacity=1)
                        st.plotly_chart(fig_line, use_container_width=True)
                        
                        total_ret = port_val.iloc[-1] - 1
                        cagr = (port_val.iloc[-1])**(1/years) - 1
                        mdd = calculate_mdd(port_val) # 計算 MDD
                        
                        st.markdown("### 💰 實際回測結果")
                        # ★修改點：分成 3 欄，加入 MDD
                        cb1, cb2, cb3 = st.columns(3)
                        cb1.metric("期間總報酬率", f"{total_ret:.2%}")
                        cb2.metric("年化報酬 (CAGR)", f"{cagr:.2%}")
                        cb3.metric("最大回撤 (MDD)", f"{mdd:.2%}", delta="越小越好", delta_color="inverse", help="歷史上最慘曾經跌掉多少")

                # --- Tab 2: 最大夏普 ---
                with tab2:
                    st.subheader("🚀 策略目標：最高 CP 值")
                    
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
                        c_s1.metric("預期年化報酬", f"{exp_ret_sharpe:.2%}")
                        c_s2.metric("預期年化波動", f"{exp_vol_sharpe:.2%}")
                        st.metric("夏普值 (CP值)", f"{sharpe_ratio:.2f}", delta="優異")
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
                        port_val_s = (normalized_prices * w_sharpe).sum(axis=1)
                        port_val_s.name = "🚀 最大夏普組合"
                        combined_s = normalized_prices.copy()
                        combined_s["🚀 最大夏普組合"] = port_val_s
                        
                        fig_line_s = px.line(combined_s, title=f'資產成長回測 (過去 {years} 年)')
                        fig_line_s.update_traces(line=dict(width=1), opacity=0.3)
                        fig_line_s.update_traces(selector=dict(name="🚀 最大夏普組合"), line=dict(color='red', width=4), opacity=1)
                        st.plotly_chart(fig_line_s, use_container_width=True)
                        
                        total_ret_s = port_val_s.iloc[-1] - 1
                        cagr_s = (port_val_s.iloc[-1])**(1/years) - 1
                        mdd_s = calculate_mdd(port_val_s)
                        
                        st.markdown("### 💰 實際回測結果")
                        csb1, csb2, csb3 = st.columns(3)
                        csb1.metric("期間總報酬率", f"{total_ret_s:.2%}")
                        csb2.metric("年化報酬 (CAGR)", f"{cagr_s:.2%}")
                        csb3.metric("最大回撤 (MDD)", f"{mdd_s:.2%}", delta="越小越好", delta_color="inverse")

                # ==========================
                # C. 進階分析：效率前緣 + 熱力圖
                # ==========================
                st.markdown("---")
                with st.expander("📊 進階分析：效率前緣與相關性", expanded=True):
                    
                    ec1, ec2 = st.columns(2)
                    
                    # 1. 效率前緣散佈圖 (新功能)
                    with ec1:
                        st.markdown("#### 效率前緣 (Efficient Frontier)")
                        st.caption("藍點：模擬的隨機組合 | 紅星：最大夏普 | 綠星：最小風險")
                        
                        fig_ef = px.scatter(df_mc, x="波動度", y="報酬率", color="夏普值", 
                                            color_continuous_scale='Viridis', hover_data=['夏普值'])
                        
                        # 標出我們算出的兩個最佳點
                        fig_ef.add_trace(go.Scatter(x=[exp_vol_min], y=[exp_ret_min], mode='markers', 
                                                    marker=dict(size=15, color='green', symbol='star'), name='最小風險'))
                        fig_ef.add_trace(go.Scatter(x=[exp_vol_sharpe], y=[exp_ret_sharpe], mode='markers', 
                                                    marker=dict(size=15, color='red', symbol='star'), name='最大夏普'))
                        st.plotly_chart(fig_ef, use_container_width=True)

                    # 2. 相關係數熱力圖
                    with ec2:
                        st.markdown("#### 資產相關性 (Correlation)")
                        st.caption("顏色越藍代表互補性越好 (風險分散效果佳)")
                        fig_corr = px.imshow(corr_matrix, text_auto='.2f', aspect="auto", 
                                             color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
                        st.plotly_chart(fig_corr, use_container_width=True)

            except Exception as e:
                st.error(f"發生錯誤：{str(e)}")
else:
    st.info("請在左側輸入股票代號並按下「開始計算」")

st.sidebar.markdown("---")
st.sidebar.caption("⚠️ **免責聲明**")
st.sidebar.caption("""
本工具僅供市場分析與模擬參考，不構成任何投資建議或邀約。
歷史績效不代表未來獲利保證。
投資人應審慎評估風險，並自負盈虧。
""")
