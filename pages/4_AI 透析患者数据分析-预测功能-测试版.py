import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
import google.generativeai as genai
import anthropic
import time
import logging
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(layout="centered", page_title="AI 透析患者数据分析-预测功能-测试版")

# Access API keys from secrets
openai_api_key = st.secrets["api_keys"]["openai_api_key"]
gemini_api_key = st.secrets["api_keys"]["gemini_api_key"]
anthropic_api_key = st.secrets["api_keys"]["anthropic_api_key"]

@st.cache_data
# Function definitions
def generate_anthropic_insights(prompt, expert_role):
    try:
        client = anthropic.Anthropic(api_key=anthropic_api_key)
        system_message = system_message_template.format(expert_role=expert_role)
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=3000,
            temperature=0,
            top_p=1,
            system=system_message,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return message.content[0].text
    except Exception as e:
        return f"Anthropic API 出错: {str(e)}. 跳过此分析。"

@st.cache_data
def generate_gpt_insights(prompt, expert_role):
    try:
        client = OpenAI(api_key=openai_api_key)
        model_type = "gpt-4o"
        system_message = system_message_template.format(expert_role=expert_role)
        completion = client.chat.completions.create(
            model=model_type,
            messages=[
                {
                    "role": "system", 
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=3000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return model_type, completion.choices[0].message.content
    except Exception as e:
        return "gpt-4o", f"OpenAI API 出错: {str(e)}. 跳过此分析。"

@st.cache_data
def generate_gemini_insights(prompt, expert_role):
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        
        insights = ""
        if response.parts:
            for part in response.parts:
                if hasattr(part, 'text'):
                    insights += part.text
        
        if not insights:
            insights = "目前无法生成详细的分析报告。"
        
        return insights
    except Exception as e:
        return f"Gemini API 出错: {str(e)}. 跳过此分析。"

@st.cache_data
def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

@st.cache_data
def load_csv_data(file_path):
    return pd.read_csv(file_path)

@st.cache_data
def analyze_dialysis_data(treatment_data):
    treatment_data['Date'] = pd.to_datetime(treatment_data['Date'])
    
    X = treatment_data[['Pre-Weight', 'Post-Weight', 'Weight Change']]
    y = treatment_data['Dry Weight']
    model = LinearRegression()
    model.fit(X, y)
    
    last_treatment = X.iloc[-1]
    predicted_dry_weight = model.predict([last_treatment])[0]
    
    avg_weight_gain = treatment_data['Weight Gain'].mean()
    suggested_uf = avg_weight_gain
    
    def bp_drop(row):
        pre_systolic = int(row['Pre-BP'].split('/')[0])
        post_systolic = int(row['Post-BP'].split('/')[0])
        return (pre_systolic - post_systolic) / pre_systolic * 100

    treatment_data['BP_Drop'] = treatment_data.apply(bp_drop, axis=1)
    avg_bp_drop = treatment_data['BP_Drop'].mean()
    std_bp_drop = treatment_data['BP_Drop'].std()
    
    def low_bp_risk(bp_drop):
        if bp_drop > avg_bp_drop + 2*std_bp_drop:
            return "高"
        elif bp_drop > avg_bp_drop + std_bp_drop:
            return "中"
        else:
            return "低"
    
    last_bp_drop = treatment_data['BP_Drop'].iloc[-1]
    low_bp_risk_next = low_bp_risk(last_bp_drop)
    low_bp_risk_percentage = (treatment_data['BP_Drop'] > avg_bp_drop + std_bp_drop).mean() * 100
    
    avg_blood_flow = treatment_data['Blood Flow'].mean()
    avg_dialysate_flow = treatment_data['Dialysate Flow'].mean()
    avg_hco3 = treatment_data['HCO3'].mean()
    
    return {
        'predicted_dry_weight': round(predicted_dry_weight, 2),
        'suggested_uf': round(suggested_uf, 2),
        'low_bp_risk_next': low_bp_risk_next,
        'low_bp_risk_percentage': round(low_bp_risk_percentage, 2),
        'avg_blood_flow': round(avg_blood_flow, 2),
        'avg_dialysate_flow': round(avg_dialysate_flow, 2),
        'avg_hco3': round(avg_hco3, 2),
        'bp_drop_data': treatment_data[['Date', 'BP_Drop']]
    }

def create_plots(treatment_data, analysis_results):
    treatment_data['Date'] = pd.to_datetime(treatment_data['Date'])
    
    fig = make_subplots(rows=3, cols=2, subplot_titles=(
        '体重趋势', '血压趋势', '超滤趋势', '血流和透析液流量趋势', '血压下降百分比', '碳酸氢根水平'
    ))

    # Plot 1: Weight trends
    fig.add_trace(go.Scatter(x=treatment_data['Date'], y=treatment_data['Pre-Weight'], name='透析前体重'), row=1, col=1)
    fig.add_trace(go.Scatter(x=treatment_data['Date'], y=treatment_data['Post-Weight'], name='透析后体重'), row=1, col=1)
    fig.add_trace(go.Scatter(x=treatment_data['Date'], y=treatment_data['Dry Weight'], name='干体重'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[treatment_data['Date'].iloc[-1]], y=[analysis_results['predicted_dry_weight']], 
                             mode='markers', name='预测干体重', marker=dict(size=10, symbol='star')), row=1, col=1)

    # Plot 2: Blood Pressure trends
    fig.add_trace(go.Scatter(x=treatment_data['Date'], y=treatment_data['Pre-BP'].apply(lambda x: int(x.split('/')[0])), name='透析前收缩压'), row=1, col=2)
    fig.add_trace(go.Scatter(x=treatment_data['Date'], y=treatment_data['Pre-BP'].apply(lambda x: int(x.split('/')[1])), name='透析前舒张压'), row=1, col=2)
    fig.add_trace(go.Scatter(x=treatment_data['Date'], y=treatment_data['Post-BP'].apply(lambda x: int(x.split('/')[0])), name='透析后收缩压'), row=1, col=2)
    fig.add_trace(go.Scatter(x=treatment_data['Date'], y=treatment_data['Post-BP'].apply(lambda x: int(x.split('/')[1])), name='透析后舒张压'), row=1, col=2)

    # Plot 3: Ultrafiltration trends
    fig.add_trace(go.Scatter(x=treatment_data['Date'], y=treatment_data['Prescribed UF'], name='处方超滤量'), row=2, col=1)
    fig.add_trace(go.Scatter(x=treatment_data['Date'], y=treatment_data['Actual UF'], name='实际超滤量'), row=2, col=1)
    fig.add_trace(go.Scatter(x=[treatment_data['Date'].iloc[-1]], y=[analysis_results['suggested_uf']], 
                             mode='markers', name='建议超滤量', marker=dict(size=10, symbol='star')), row=2, col=1)

    # Plot 4: Blood Flow and Dialysate Flow trends
    fig.add_trace(go.Scatter(x=treatment_data['Date'], y=treatment_data['Blood Flow'], name='血流量'), row=2, col=2)
    fig.add_trace(go.Scatter(x=treatment_data['Date'], y=treatment_data['Dialysate Flow'], name='透析液流量'), row=2, col=2)

    # Plot 5: Blood Pressure Drop
    fig.add_trace(go.Scatter(x=analysis_results['bp_drop_data']['Date'], y=analysis_results['bp_drop_data']['BP_Drop'], name='血压下降百分比'), row=3, col=1)

    # Plot 6: HCO3 levels
    fig.add_trace(go.Scatter(x=treatment_data['Date'], y=treatment_data['HCO3'], name='碳酸氢根水平'), row=3, col=2)

    # Update layout
    fig.update_layout(
        height=1200, 
        title_text='月度透析治疗趋势及预测',
        showlegend=False,
        hovermode="x unified",
        barmode='group'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text='日期', row=1, col=1)
    fig.update_xaxes(title_text='日期', row=1, col=2)
    fig.update_xaxes(title_text='日期', row=2, col=1)
    fig.update_xaxes(title_text='日期', row=2, col=2)
    fig.update_xaxes(title_text='日期', row=3, col=1)
    fig.update_xaxes(title_text='日期', row=3, col=2)

    fig.update_yaxes(title_text='体重 (kg)', row=1, col=1)
    fig.update_yaxes(title_text='血压 (mmHg)', row=1, col=2)
    fig.update_yaxes(title_text='超滤量 (L)', row=2, col=1)
    fig.update_yaxes(title_text='流量 (mL/min)', row=2, col=2)
    fig.update_yaxes(title_text='血压下降 (%)', row=3, col=1)
    fig.update_yaxes(title_text='HCO3 (mmol/L)', row=3, col=2)

    return fig

# Load data
lab_data = load_json_data('files/lab_sxl.json')
treatment_data = load_csv_data('files/treatment_month_sxl.csv')

# System message template
system_message_template = """
你是一位经验丰富的透析和肾脏病专家，同时也是一位医生的AI助手。你的任务是分析患者的检查结果和一个月的透析治疗数据，
并提供全面的医疗见解和治疗建议。你的分析和建议应该：

1. 基于最新的临床指南和研究
2. 针对患者的具体情况提供个性化的建议
3. 使用具体的数据点来支持你的分析，特别注意月度数据的趋势和变化
4. 提供明确的、可操作的建议
5. 引用相关的医学文献或指南来支持你的建议，包括至少一个中文来源和国际指南（如KDIGO）

请按照以下结构提供你的分析和建议，并在适当的地方使用表格和图表来呈现数据：

1. 患者概况总结（包括月度治疗数据的总体趋势）
2. 主要发现和关注点（使用表格格式）
3. 详细分析（包括以下方面，并在适当时使用表格或图表）：
   a. 贫血管理
   b. 钙磷代谢
   c. 营养状况
   d. 透析充分性
   e. 血压控制和血流动力学
   f. 体重管理和水分平衡
   g. 并发症预防和管理
   h. 长期预后
4. 治疗建议和调整方案（使用表格格式）
5. 需要进一步检查或监测的项目（使用表格格式）
6. 参考文献和指南（包括中文和国际来源）

请确保你的回答具体、明确，并直接引用患者数据来支持你的分析和建议。对于表格和图表，请使用Markdown格式。
"""

def generate_structured_prompt(lab_data, treatment_data, analysis_results):
    prompt = f"""
请分析以下透析患者的检查结果和一个月的透析治疗数据，并提供详细的医疗分析和治疗建议。

患者基本信息：
- 姓名：{lab_data['患者信息']['姓名']}
- 年龄：{lab_data['患者信息']['年龄']}岁
- 性别：{lab_data['患者信息']['性别']}
- 透析号：{lab_data['患者信息']['透析号']}

检查结果：
{json.dumps(lab_data, ensure_ascii=False, indent=2)}

一个月透析治疗数据摘要：
- 平均透析前体重：{treatment_data['Pre-Weight'].mean():.2f} kg
- 平均透析后体重：{treatment_data['Post-Weight'].mean():.2f} kg
- 平均体重变化：{treatment_data['Weight Change'].mean():.2f} kg
- 平均透析前血压：{treatment_data['Pre-BP'].mode().values[0]}
- 平均透析后血压：{treatment_data['Post-BP'].mode().values[0]}
- 平均血流流量：{treatment_data['Blood Flow'].mean():.0f} mL/min
- 平均透析液流量：{treatment_data['Dialysate Flow'].mean():.0f} mL/min

预测和风险分析结果：
- 预测干体重：{analysis_results['predicted_dry_weight']} kg
- 建议超滤量：{analysis_results['suggested_uf']} L
- 下次低血压风险：{analysis_results['low_bp_risk_next']}
- 低血压风险百分比：{analysis_results['low_bp_risk_percentage']}%
- 平均血流量：{analysis_results['avg_blood_flow']} mL/min
- 平均透析液流量：{analysis_results['avg_dialysate_flow']} mL/min
- 平均碳酸氢根：{analysis_results['avg_hco3']} mmol/L

请根据上述数据和图表，按照以下结构提供你的分析和建议：

1. 患者概况总结（简要概述患者的主要情况和关键问题，包括月度治疗数据的总体趋势）

2. 主要发现和关注点（使用Markdown表格格式列出3-5个最重要的发现或问题）
å
3. 详细分析：
   对于每个方面，请：
   - 指出异常或需要关注的数据点
   - 解释这些发现的临床意义
   - 分析月度数据的趋势和变化
   - 将患者的数据与标准指南或目标值进行比较
   - 提供具体的改善建议
   - 在适当时使用Markdown表格或描述图表来呈现数据

   a. 贫血管理
   b. 钙磷代谢
   c. 营养状况
   d. 透析充分性
   e. 血压控制和血流动力学
   f. 体重管理和水分平衡
   g. 并发症预防和管理
   h. 长期预后

4. 治疗建议和调整方案（使用Markdown表格格式，包括具体的药物调整、透析处方修改、饮食和生活方式建议）

5. 需要进一步检查或监测的项目（使用Markdown表格格式）

---
参考文献和指南（请列出3-4个支持你建议的关键参考文献或指南，包括至少一个中文来源和一个KDIGO指南）

请确保你的分析和建议具体、明确，并直接引用患者数据和图表趋势来支持你的结论。你的回答应该是全面的，但也要简洁明了，便于医生快速理解和采取行动。
"""
    return prompt

# Streamlit app
st.title("AI 透析患者数据分析")

st.markdown("""
这个应用程序利用多个AI模型（GPT-4、Gemini Pro、Claude-3）来分析透析患者的医疗数据。
它提供基于最新临床指南的详细医疗分析和个性化治疗建议，包括一个月的透析治疗数据和趋势图表。
""")

st.header("患者数据")
col1, col2 = st.columns(2)
with col1:
    with st.expander("检查结果"):
        st.json(lab_data)
with col2:
    with st.expander("透析治疗数据（一个月）"):
        st.dataframe(treatment_data)

# Analyze data and create plots
analysis_results = analyze_dialysis_data(treatment_data)
fig = create_plots(treatment_data, analysis_results)

# Display plots
with st.expander("月度数据趋势图表及预测"):
    st.plotly_chart(fig, use_container_width=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("预测干体重", f"{analysis_results['predicted_dry_weight']} kg")
        st.metric("建议超滤量", f"{analysis_results['suggested_uf']} L")
    with col2:
        st.metric("下次低血压风险", analysis_results['low_bp_risk_next'])
        st.metric("低血压风险百分比", f"{analysis_results['low_bp_risk_percentage']}%")
    with col3:
        st.metric("平均血流量", f"{analysis_results['avg_blood_flow']} mL/min")
        st.metric("平均透析液流量", f"{analysis_results['avg_dialysate_flow']} mL/min")
        st.metric("平均碳酸氢根", f"{analysis_results['avg_hco3']} mmol/L")

st.header("AI 分析结果")
if st.button('开始分析'):
    prompt = generate_structured_prompt(lab_data, treatment_data, analysis_results)

    # Progress bar
    progress_bar = st.progress(0)
    total_analyses = 3  # GPT, Gemini, Claude
    progress_step = 1 / total_analyses

    tab1, tab2, tab3 = st.tabs(["GPT-4", "Gemini Pro", "Claude-3"])

    with tab1:
        st.subheader("GPT-4 医疗分析")
        with st.spinner('GPT-4 分析中，请稍候...'):
            start_time_gpt = time.time()
            model_type, gpt_insights = generate_gpt_insights(prompt, "透析和肾脏病专家")
            end_time_gpt = time.time()
            st.info(f"GPT-4 分析耗时: {end_time_gpt - start_time_gpt:.2f} 秒")
            st.markdown(gpt_insights)
            progress_bar.progress(1 * progress_step)

    with tab2:
        st.subheader("Gemini Pro 医疗分析")
        with st.spinner('Gemini Pro 分析中，请稍候...'):    
            start_time_gemini = time.time()
            gemini_insights = generate_gemini_insights(prompt, "透析和肾脏病专家")
            end_time_gemini = time.time()
            st.info(f"Gemini Pro 分析耗时: {end_time_gemini - start_time_gemini:.2f} 秒")
            st.markdown(gemini_insights)
            progress_bar.progress(2 * progress_step)

    with tab3:
        st.subheader("Claude-3 医疗分析")
        with st.spinner('Claude-3 分析中，请稍候...'):
            start_time_claude = time.time()
            claude_insights = generate_anthropic_insights(prompt, "透析和肾脏病专家")
            end_time_claude = time.time()
            st.info(f"Claude-3 分析耗时: {end_time_claude - start_time_claude:.2f} 秒")
            st.markdown(claude_insights)
            progress_bar.progress(3 * progress_step)

else:
    st.info("点击'开始分析'以开始。")

st.markdown("---")
st.markdown("© 2024 AI 透析患者数据分析应用. All rights reserved.")