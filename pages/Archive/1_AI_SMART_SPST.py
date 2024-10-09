import streamlit as st
import json
from openai import OpenAI
import google.generativeai as genai
import anthropic
import requests
import logging
import pandas as pd
import matplotlib.pyplot as plt
import time

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
        # Initialize the OpenAI client with the api_key
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
            temperature=0,  # Reduce randomness for consistent output
            max_tokens=3000,  # Ensure enough tokens for long responses
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return model_type, completion.choices[0].message.content
    except Exception as e:
        return "gpt-4o", f"OpenAI API 出错: {str(e)}. 跳过此分析。"

logging.basicConfig(level=logging.INFO)

def generate_gemini_insights(prompt, expert_role):
    try:
        # Set up the API key
        genai.configure(api_key=gemini_api_key)
        # Prepare the data summary
        
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

# Function to load JSON data
def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Load JSON data
lab_data = load_json_data('files/lab_sxl.json')
treatment_data = load_json_data('files/treatment_sxl.json')

# Updated system_message_template
system_message_template = """
你是一位经验丰富的透析和肾脏病专家，同时也是一位医生的AI助手。你的任务是分析患者的检查结果和透析治疗数据，
并提供全面的医疗见解和治疗建议。你的分析和建议应该：

1. 基于最新的临床指南和研究
2. 针对患者的具体情况提供个性化的建议
3. 使用具体的数据点来支持你的分析
4. 提供明确的、可操作的建议
5. 引用相关的医学文献或指南来支持你的建议，包括至少一个中文来源和国际指南（如KDIGO）

请按照以下结构提供你的分析和建议，并在适当的地方使用表格和图表来呈现数据：

1. 患者概况总结
2. 主要发现和关注点（使用表格格式）
3. 详细分析（包括以下方面，并在适当时使用表格或图表）：
   a. 贫血管理
   b. 钙磷代谢
   c. 营养状况
   d. 透析充分性
   e. 血压控制
   f. 并发症预防和管理
   g. 长期预后
4. 治疗建议和调整方案（使用表格格式）
5. 需要进一步检查或监测的项目（使用表格格式）
6. 参考文献和指南（包括中文和国际来源）

请确保你的回答具体、明确，并直接引用患者数据来支持你的分析和建议。对于表格和图表，请使用Markdown格式。
"""

# New function to generate a more structured prompt
def generate_structured_prompt(lab_data, treatment_data):
    prompt = f"""
请分析以下透析患者的检查结果和透析治疗数据，并提供详细的医疗分析和治疗建议。

患者基本信息：
- 姓名：{lab_data['患者信息']['姓名']}
- 年龄：{lab_data['患者信息']['年龄']}岁
- 性别：{lab_data['患者信息']['性别']}
- 首次透析时间：{treatment_data['患者信息']['首次透析时间']}

检查结果：
{json.dumps(lab_data, ensure_ascii=False, indent=2)}

透析治疗数据：
{json.dumps(treatment_data, ensure_ascii=False, indent=2)}

请按照以下结构提供你的分析和建议，并在适当的地方使用表格和图表：

1. 患者概况总结（简要概述患者的主要情况和关键问题）

2. 主要发现和关注点（使用Markdown表格格式列出3-5个最重要的发现或问题）

3. 详细分析：
   对于每个方面，请：
   - 指出异常或需要关注的数据点
   - 解释这些发现的临床意义
   - 将患者的数据与标准指南或目标值进行比较
   - 提供具体的改善建议
   - 在适当时使用Markdown表格或描述图表来呈现数据

   a. 贫血管理
   b. 钙磷代谢
   c. 营养状况
   d. 透析充分性
   e. 血压控制
   f. 并发症预防和管理
   g. 长期预后

4. 治疗建议和调整方案（使用Markdown表格格式，包括具体的药物调整、透析处方修改、饮食和生活方式建议）

5. 需要进一步检查或监测的项目（使用Markdown表格格式）

---
参考文献和指南（请列出3-4个支持你建议的关键参考文献或指南，包括至少一个中文来源和一个KDIGO指南）

请确保你的分析和建议具体、明确，并直接引用患者数据来支持你的结论。你的回答应该是全面的，但也要简洁明了，便于医生快速理解和采取行动。对于表格，请使用Markdown格式；对于图表，请描述你建议的图表内容，我们的系统会根据你的描述生成相应的图表。
"""
    return prompt


# Page setup
st.set_page_config(layout="centered", page_title="AI 透析患者数据分析")
st.title("AI 透析患者数据分析")

# Add application description
st.markdown("""
这个应用程序利用多个AI模型（GPT-4、Gemini Pro、Claude-3和Llama-3.1）来分析透析患者的医疗数据。
它提供基于最新临床指南的详细医疗分析和个性化治疗建议。
""")

# Display patient data
st.header("患者数据")
col1, col2 = st.columns(2)
with col1:
    with st.expander("检查结果"):
        st.json(lab_data)
with col2:
    with st.expander("透析治疗数据"):
        st.json(treatment_data)

# Analysis section
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
            start_time_gpt = time.time()  # Start time for GPT-4
            model_type, gpt_insights = generate_gpt_insights(prompt, "透析和肾脏病专家")
            end_time_gpt = time.time()  # End time for GPT-4
            st.info(f"GPT-4 分析耗时: {end_time_gpt - start_time_gpt:.2f} 秒")
            st.write(gpt_insights)
            progress_bar.progress(1 * progress_step)

    with tab2:
        st.subheader("Gemini Pro 医疗分析")
        with st.spinner('Gemini Pro 分析中，请稍候...'):    
            start_time_gemini = time.time()  # Start time for Gemini Pro
            gemini_insights = generate_gemini_insights(prompt, "透析和肾脏病专家")
            end_time_gemini = time.time()  # End time for Gemini Pro
            st.info(f"Gemini Pro 分析耗时: {end_time_gemini - start_time_gemini:.2f} 秒")
            st.write(gemini_insights)
            progress_bar.progress(2 * progress_step)

    with tab3:
        st.subheader("Claude-3 医疗分析")
        with st.spinner('Claude-3 分析中，请稍候...'):
            start_time_claude = time.time()  # Start time for Claude-3
            claude_insights = generate_anthropic_insights(prompt, "透析和肾脏病专家")
            end_time_claude = time.time()  # End time for Claude-3
            st.info(f"Claude-3 分析耗时: {end_time_claude - start_time_claude:.2f} 秒")
            st.write(claude_insights)
            progress_bar.progress(3 * progress_step)

else:
    st.info("点击'开始分析'以开始。")