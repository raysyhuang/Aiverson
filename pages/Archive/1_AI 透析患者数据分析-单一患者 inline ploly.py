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

# Access API keys from secrets
openai_api_key = st.secrets["api_keys"]["openai_api_key"]
gemini_api_key = st.secrets["api_keys"]["gemini_api_key"]
anthropic_api_key = st.secrets["api_keys"]["anthropic_api_key"]

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

def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_csv_data(file_path):
    return pd.read_csv(file_path)

def create_plots(treatment_data):
    treatment_data['Date'] = pd.to_datetime(treatment_data['Date'])
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        '体重趋势', '血压趋势', '超滤趋势', '血流和透析液流量趋势'
    ))

    # Plot 1: Weight trends
    fig.add_trace(go.Scatter(x=treatment_data['Date'], y=treatment_data['Pre-Weight'], name='透析前体重'), row=1, col=1)
    fig.add_trace(go.Scatter(x=treatment_data['Date'], y=treatment_data['Post-Weight'], name='透析后体重'), row=1, col=1)
    fig.add_trace(go.Scatter(x=treatment_data['Date'], y=treatment_data['Dry Weight'], name='干体重'), row=1, col=1)

    # Plot 2: Blood Pressure trends
    fig.add_trace(go.Scatter(x=treatment_data['Date'], y=treatment_data['Pre-BP'].apply(lambda x: int(x.split('/')[0])), name='透析前收缩压'), row=1, col=2)
    fig.add_trace(go.Scatter(x=treatment_data['Date'], y=treatment_data['Pre-BP'].apply(lambda x: int(x.split('/')[1])), name='透析前舒张压'), row=1, col=2)
    fig.add_trace(go.Scatter(x=treatment_data['Date'], y=treatment_data['Post-BP'].apply(lambda x: int(x.split('/')[0])), name='透析后收缩压'), row=1, col=2)
    fig.add_trace(go.Scatter(x=treatment_data['Date'], y=treatment_data['Post-BP'].apply(lambda x: int(x.split('/')[1])), name='透析后舒张压'), row=1, col=2)

    # Plot 3: Ultrafiltration trends
    fig.add_trace(go.Scatter(x=treatment_data['Date'], y=treatment_data['Prescribed UF'], name='处方超滤量'), row=2, col=1)
    fig.add_trace(go.Scatter(x=treatment_data['Date'], y=treatment_data['Actual UF'], name='实际超滤量'), row=2, col=1)

    # Plot 4: Blood Flow and Dialysate Flow trends
    fig.add_trace(go.Scatter(x=treatment_data['Date'], y=treatment_data['Blood Flow'], name='血流量'), row=2, col=2)
    fig.add_trace(go.Scatter(x=treatment_data['Date'], y=treatment_data['Dialysate Flow'], name='透析液流量'), row=2, col=2)

    # Update layout
    fig.update_layout(height=800, 
                      showlegend=False, 
                      title_text='月度透析治疗趋势',
                      hovermode="x unified",
                      barmode='group'
                      )

    # Update axes labels
    fig.update_xaxes(title_text='日期', row=1, col=1)
    fig.update_xaxes(title_text='日期', row=1, col=2)
    fig.update_xaxes(title_text='日期', row=2, col=1)
    fig.update_xaxes(title_text='日期', row=2, col=2)

    fig.update_yaxes(title_text='体重 (kg)', row=1, col=1)
    fig.update_yaxes(title_text='血压 (mmHg)', row=1, col=2)
    fig.update_yaxes(title_text='超滤量 (L)', row=2, col=1)
    fig.update_yaxes(title_text='流量 (mL/min)', row=2, col=2)

    return fig

# Load data from single files
all_lab_data = load_json_data('files/lab_month_3.json')
all_treatment_data = load_csv_data('files/treatment_month_3.csv')

# Extract patient list
patients = [patient['患者信息']['姓名'] for patient in all_lab_data if '患者信息' in patient]

# System message template
system_message_template = """
你是一位经验丰富的透析和肾脏病专家，同时也是一位医生的AI助手。你的任务是全面分析患者的检查结果和一个月的透析治疗数据，并提供详细的医疗见解和个性化的治疗建议。你的分析和建议应该：
1 基于最新的临床指南和研究，如KDIGO指南和中国相关指南。
2 针对患者的具体情况，提供个性化和可操作的建议。
3 逐一分析所有关键数据点，包括正常和异常值，并特别注意月度数据的趋势和变化。
4 解释每个发现的临床意义，并将患者的数据与标准目标值进行详细比较。
5 提供具体的改善建议，包括药物调整、透析处方修改、饮食和生活方式建议。
6 引用相关的医学文献或指南来支持你的建议，包括至少一个中文来源和国际指南（如KDIGO）。

请按照以下结构提供你的分析和建议，并在适当的地方使用Markdown格式的表格和图表来呈现数据：
1. 患者概况总结
* 简要概述患者的主要情况和关键问题。
* 包括月度治疗数据的总体趋势和任何显著变化。

2. 主要发现和关注点
* 使用Markdown表格列出3-5个最重要的发现或问题。
* 每个发现应包括数据点、临床意义和需要关注的原因。

3. 详细分析
* 对于每个方面（见下方列表），请：
  * 逐一指出异常或需要关注的数据点，并包含正常参考范围。
  * 解释这些发现的临床意义，以及可能的原因和影响。
  * 分析月度数据的趋势和变化，使用图表或表格展示（如适用）。
  * 将患者的数据与标准指南或目标值进行详细比较，引用具体的指南和目标范围。
  * 提供具体的改善建议，包括可采取的措施和预期目标。
* 需要分析的方面： a. 贫血管理 b. 钙磷代谢 c. 营养状况 d. 透析充分性 e. 血压控制和血流动力学 f. 体重管理和水分平衡 g. 并发症预防和管理 h. 长期预后

4. 治疗建议和调整方案
* 使用Markdown表格列出具体的治疗建议。
* 包括药物调整（剂量、频率）、透析处方修改、饮食和生活方式建议。

5. 需要进一步检查或监测的项目
* 使用Markdown表格列出需要额外关注的检查或监测项目。
* 每项应包括检查目的和建议的频率。

6. 参考文献和指南
* 列出3-4个支持你建议的关键参考文献或指南。
* 包括至少一个中文来源和一个KDIGO指南，并注明具体章节或页码。

请确保：
* 你的回答具体、明确，直接引用患者数据和图表趋势来支持你的结论。
* 避免遗漏任何重要数据或异常值，即使是正常值也应考虑其趋势。
* 回答应全面且简洁明了，便于医生快速理解和采取行动。
"""

def generate_structured_prompt(lab_data, treatment_data):
    prompt = f"""
请分析以下透析患者的检查结果和一个月的透析治疗数据，并提供详细的医疗分析和治疗建议。
患者基本信息：
* 姓名：{lab_data['患者信息']['姓名']}
* 年龄：{lab_data['患者信息']['年龄']}岁
* 性别：{lab_data['患者信息']['性别']}

检查结果：
{json.dumps(lab_data, ensure_ascii=False, indent=2)}

透析治疗数据（一个月）：
{treatment_data.to_markdown(index=False)}

请按以下结构提供您的分析和建议：

1. 患者概况
   - 总结患者的整体状况和主要问题。
   - 强调月度数据中的显著趋势或变化。

2. 主要发现
   | 发现 | 数据/趋势 | 临床意义 |
   |------|-----------|----------|
   | [发现1] | [数据1] | [意义1] |
   | [发现2] | [数据2] | [意义2] |
   | [发现3] | [数据3] | [意义3] |

3. 详细分析
   对于每个领域（a-h），请提供：
   - 异常或需关注的数据点及其参考范围
   - 临床解释，包括潜在原因和影响
   - 与指南目标的比较
   - 具体改进建议
   - 使用Plotly创建图表的Python代码，展示该领域相关指标的月度进展
   
   需分析的领域：
   a. 贫血管理 
   b. 矿物质和骨病变 
   c. 营养状况 
   d. 透析充分性 
   e. 血压和血流动力学 
   f. 液体管理 
   g. 并发症预防
   h. 长期预后

4. 治疗建议
   | 方面 | 建议 | 目标 |
   |------|------|------|
   | [方面1] | [建议1] | [目标1] |
   | [方面2] | [建议2] | [目标2] |
   | [方面3] | [建议3] | [目标3] |

5. 后续跟进建议
   | 检查/监测项目 | 目的 | 频率 |
   |---------------|------|------|
   | [检查1] | [目的1] | [频率1] |
   | [检查2] | [目的2] | [频率2] |

6. 参考文献
   - [参考文献1]：[简短描述]
   - [参考文献2]：[简短描述]
   - [参考文献3]：[简短描述]

确保您的分析基于证据、针对患者具体情况，并且可操作。在您的建议中考虑生活质量和心理社会因素。为每个分析领域创建并包含相关的Plotly图表代码，展示月度进展。
"""
    return prompt

# Streamlit app
st.set_page_config(layout="centered", page_title="AI 透析患者数据分析")
st.title("AI 透析患者数据分析")

st.markdown("""
这个应用程序利用多个AI模型（GPT-4、Gemini Pro、Claude-3）来分析透析患者的医疗数据。
它提供基于最新临床指南的详细医疗分析和个性化治疗建议，包括一个月的透析治疗数据和趋势图表。
""")

# Convert list to dictionary
all_lab_data_dict = {patient['患者信息']['姓名']: patient for patient in all_lab_data}

# Patient selection
st.header("患者选择")
selected_patient = st.selectbox("选择要分析的患者", patients)

if selected_patient:
    st.header("患者数据")
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("检查结果"):
            st.json(all_lab_data_dict[selected_patient])
    with col2:
        with st.expander("透析治疗数据（一个月）"):
            patient_treatment_data = all_treatment_data[all_treatment_data['Patient'] == selected_patient].reset_index(drop=True)
            st.dataframe(patient_treatment_data)

    # Display plots
    with st.expander("月度数据趋势图表"):
        fig = create_plots(patient_treatment_data)
        st.plotly_chart(fig, use_container_width=True)

    # Generate the prompt for the selected patient
    st.header("AI 分析结果")
    if st.button('开始分析'):
        prompt = generate_structured_prompt(all_lab_data_dict[selected_patient], patient_treatment_data)

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