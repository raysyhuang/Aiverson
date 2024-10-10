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
import statistics

# Set page configuration
st.set_page_config(layout="centered", page_title="AI 透析患者数据分析-多名患者")

# Access API keys from secrets
openai_api_key = st.secrets["api_keys"]["openai_api_key"]
gemini_api_key = st.secrets["api_keys"]["gemini_api_key"]
anthropic_api_key = st.secrets["api_keys"]["anthropic_api_key"]

# Function definitions
@st.cache_data
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
        return f"Anthropic API error: {str(e)}. Skipping this analysis."

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
        return "gpt-4o", f"OpenAI API error: {str(e)}. Skipping this analysis."

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
            insights = "Unable to generate a detailed analysis report at this time."
        
        return insights
    except Exception as e:
        return f"Gemini API error: {str(e)}. Skipping this analysis."

@st.cache_data
def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

@st.cache_data
def load_csv_data(file_path):
    return pd.read_csv(file_path)

# Load and preprocess data
lab_data = load_json_data('files/lab_month_3.json')
treatment_data = load_csv_data('files/treatment_month_3.csv')


# ... The System and Prompt ...

# 系统消息模板
system_message_template = """
您是一位专门从事透析中心分析和肾脏病学的高级 AI 助手。您的任务是分析透析中心在给定期间内多名患者的数据。请提供全面摘要，识别关键问题，并为个别患者和整个中心提出解决方案。您的分析应：

1. 遵循当前的临床指南，包括 KDIGO 和相关的中国指南。
2. 识别个体和汇总数据中的显著趋势、异常值和模式。
3. 优先考虑需要即时关注或长期管理的临床相关发现。
4. 提供基于证据的建议，以改善患者护理和中心协议。

请按以下结构组织您的分析，使用 Markdown 格式来增强可读性：

1. 中心概览
   * 总结关键统计数据（患者数量、人口统计、治疗类型）
   * 突出任何中心范围内的重要发现或趋势
   * 使用 Markdown 表格呈现概览数据

2. 关键指标分析
   * 使用描述性统计和 Markdown 表格分析以下方面：
     a. 贫血管理（如血红蛋白、铁蛋白）
     b. 矿物质和骨骼异常（如钙、磷、PTH）
     c. 营养状况（如白蛋白、BMI）
     d. 透析充分性（如Kt/V、尿素清除率）
     e. 血压控制
     f. 体液管理（如体重增加、超滤量）
   * 标识异常值和需要关注的患者
   * 使用 Markdown 格式的列表突出显示关键发现

3. 患者分组分析
   * 基于关键指标或结果将患者分组
   * 识别每个组的共同特征和独特挑战
   * 使用 Markdown 表格展示分组结果

4. 个体患者警报
   * 列出需要立即关注的患者
   * 使用 Markdown 表格简要说明每位患者的主要问题和建议干预措施

5. 治疗模式分析
   * 比较不同透析模式（如HD vs HDF）的效果
   * 分析透析处方参数与患者结果的关系
   * 使用 Markdown 表格呈现比较结果

6. 趋势分析
   * 识别随时间变化的重要趋势
   * 突出显示任何周期性模式或季节性影响
   * 使用 Markdown 格式的列表或表格呈现主要趋势

7. 改进建议
   * 提出具体的、可操作的建议来改善：
     a. 个体患者护理
     b. 中心范围的协议
     c. 资源分配
     d. 员工培训
   * 使用 Markdown 格式的列表或表格呈现建议

---

参考文献
   * 列出支持您分析和建议的关键指南和研究
   * 包括相关的中国和国际来源
   * 使用 Markdown 格式的参考文献列表

确保您的回应：
* 统计上合理，使用适当的方法来识别显著趋势和异常值
* 临床相关，专注于对患者护理有实际影响的发现
* 可操作，提供明确的后续步骤和建议
* 全面，涵盖透析护理的所有关键方面
* 个性化，针对特定患者群体和中心的独特需求
* 具体、明确，直接引用患者数据和表格趋势来支持你的结论
* 涵盖所有关键数据点，包括正常和异常值
* 遵循最新的临床指南，并在比较时引用具体的标准和目标范围
* 提供可操作的建议，便于医生快速理解和采取行动
* 使用Markdown格式来呈现表格，确保清晰易读
"""

def generate_structured_prompt(lab_data, treatment_data):
    # 计算基本统计数据
    total_patients = len(set(treatment_data['Patient']))
    patient_ages = [patient['患者信息']['年龄'] for patient in lab_data]
    gender_distribution = {patient['患者信息']['性别']: 0 for patient in lab_data}
    for patient in lab_data:
        gender_distribution[patient['患者信息']['性别']] += 1

    # 计算关键指标的描述性统计
    hemoglobin_values = [patient['血常规']['血红蛋白'] for patient in lab_data]
    albumin_values = [patient['肝功能']['白蛋白'] for patient in lab_data]
    phosphorus_values = [patient['电解质']['无机磷'] for patient in lab_data]
    calcium_values = [patient['电解质']['钙'] for patient in lab_data]
    potassium_values = [patient['电解质']['钾'] for patient in lab_data]

    # 创建 Markdown 表格
    overview_table = f"""
| 指标 | 值 |
|------|-----|
| 总患者数 | {total_patients} |
| 年龄范围 | {min(patient_ages)} - {max(patient_ages)} 岁 |
| 性别分布 | 男 {gender_distribution.get('男', 0)}，女 {gender_distribution.get('女', 0)} |
"""

    key_metrics_table = f"""
| 指标 | 范围 | 中位数 |
|------|------|--------|
| 血红蛋白 (g/L) | {min(hemoglobin_values):.1f} - {max(hemoglobin_values):.1f} | {statistics.median(hemoglobin_values):.1f} |
| 白蛋白 (g/L) | {min(albumin_values):.1f} - {max(albumin_values):.1f} | {statistics.median(albumin_values):.1f} |
| 磷 (mmol/L) | {min(phosphorus_values):.2f} - {max(phosphorus_values):.2f} | {statistics.median(phosphorus_values):.2f} |
| 钙 (mmol/L) | {min(calcium_values):.2f} - {max(calcium_values):.2f} | {statistics.median(calcium_values):.2f} |
| 钾 (mmol/L) | {min(potassium_values):.2f} - {max(potassium_values):.2f} | {statistics.median(potassium_values):.2f} |
"""

    prompt = f"""
请分析以下来自透析中心的数据，涵盖{total_patients}名患者。提供全面摘要，识别关键问题，并为个别患者和整个中心提出解决方案。

实验室数据：
```json
{json.dumps(lab_data, ensure_ascii=False, indent=2)}
```

治疗数据：
{treatment_data.to_markdown(index=False)}

中心概览：
{overview_table}

关键指标统计：
{key_metrics_table}

请根据这些数据提供您的分析和建议，遵循系统消息中概述的结构。重点关注：

1. 识别关键的异常值和趋势
2. 对需要立即关注的患者进行分类和优先排序
3. 分析不同治疗模式（HD vs HDF）的效果
4. 提出改善个体患者护理和中心整体表现的具体建议

您的分析应该既全面又简洁，重点突出最重要的发现和最紧迫的行动项目。请使用 Markdown 格式来增强您回复的可读性和结构性。
"""
    return prompt



# ... The App ...

# Streamlit app
st.title("AI 透析患者数据分析 - 多名患者")

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

st.header("AI 分析结果")
if st.button('开始分析'):
    prompt = generate_structured_prompt(lab_data, treatment_data)

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