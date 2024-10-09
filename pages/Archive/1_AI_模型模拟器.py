from openai import OpenAI
import streamlit as st
import pandas as pd
import google.generativeai as genai
import anthropic
import os
from files.medical_data import medical_data_ckd, medical_data_dialysis, dialysis_prompt, CKD_prompt, system_message_template  # Import the data and prompts
import requests
import json
import logging
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
        # Extract the text content from the TextBlock
        if isinstance(message.content, list) and len(message.content) > 0:
            text_block = message.content[0]
            if hasattr(text_block, 'text'):
                return text_block.text
            elif isinstance(text_block, str):
                # If it's a string representation of TextBlock, extract the text content
                return text_block.split("text='", 1)[1].split("', type='text')", 1)[0]
        return "无法提取文本内容"
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

def generate_llama_insights(prompt, expert_role):
    try:
        url = "http://localhost:1234/v1/chat/completions"
        headers = {
            "Content-Type": "application/json"
        }
        system_message = system_message_template.format(expert_role=expert_role)
        data = {
            "model": "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0
        }
        
        logging.info("Attempting to connect to Llama server...")
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        result = response.json()
        logging.info("Successfully connected and received response from Llama server.")
        return result['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error: {e}")
        return f"Llama-3.1 连接错误: {str(e)}. 请确保本地Llama服务器正在运行并可访问。"
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return f"Llama-3.1 出错: {str(e)}. 错误类型: {type(e).__name__}. 跳过此分析。"

def test_llama_connection():
    try:
        url = "http://localhost:1234/v1/models"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        st.success("Successfully connected to Llama server.")
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to Llama server: {e}")
        return False


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


# Example default values or values retrieved from data
ESRD_risk_percentage = 1  # Placeholder percentage for ESRD risk
eGFR_value = 1  # Example value for eGFR in mL/min/1.73 m²
ACR_value = 1  # Example value for ACR in mg/g
age = 1  # Example patient age
gender = "男性"  # Example gender
systolic_bp = 1  # Example systolic blood pressure
hb_value = 1  # Example hemoglobin level

# Define prompts for CKD using CKD_prompt
prompt_medical_ckd = []
for medical_df in medical_data_ckd.values():
    prompt_medical_ckd.append(CKD_prompt.format(
        medical_df=medical_df,
        ESRD_risk_percentage=ESRD_risk_percentage,
        eGFR_value=eGFR_value,
        ACR_value=ACR_value,
        systolic_bp=systolic_bp,
        hb_value=hb_value
    ))

# Define prompts for dialysis using dialysis_prompt2
prompt_medical_dialysis = []
for medical_df in medical_data_dialysis.values():
    # Ensure medical_df is correctly formatted and contains line breaks
    if isinstance(medical_df, str) and '\n' in medical_df:
        prompt_medical_dialysis.append(dialysis_prompt.format(medical_df=medical_df))
    else:
        raise ValueError("medical_df is not correctly formatted or does not contain expected line breaks")


# Page setup
st.set_page_config(layout="centered", page_title="AI 数据分析")
st.title("AI 数据分析")
# 添加应用描述
st.markdown("""
这个应用程序利用多个AI模型（GPT-4、Gemini Pro和Claude-3）来分析慢性肾病（CKD）和透析患者的医疗数据。
它可以根据选择的专家角色（CKD肾病专家或透析专家）提供详细的医疗分析和建议。
用户可以查看患者数据，并获得基于最新临床指南的个性化见解和治疗建议。
""")
with st.expander('未来AI数据预测和分析的扩展方向'):
    st.write("""
    1. **个性化治疗方案**: 利用机器学习算法，基于患者的历史数据和治疗反应，生成更加个性化的治疗方案。
    2. **预测模型优化**: 整合更多维度的患者数据，如基因信息、生活方式等，提高疾病进展预测的准确性。
    3. **实时监测与预警**: 开发可穿戴设备接口，实时监测患者状况，并在出现异常时及时预警。
    4. **多模态数据分析**: 结合影像学数据、病理数据等多模态信息，提供更全面的诊断支持。
    5. **自然语言处理升级**: 改进NLP能力，更好地理解和提取电子病历中的非结构化数据。
    6. **跨学科AI模型**: 开发能够整合肾脏病学、心脏病学、内分泌学等多个领域知识的综合AI模型。
    7. **可解释AI**: 增强AI决策的可解释性，帮助医生和患者更好地理解AI给出的建议和预测。
    8. **联邦学习应用**: 在保护患者隐私的前提下，通过联邦学习技术整合多家医疗机构的数据，提升模型性能。
    """)


# Input section
st.header("输入参数")
col1, col2 = st.columns(2)
with col1:
    expert_role = st.selectbox("选择专家角色", ["CKD慢性肾病专家", "血液透析专家"])
    # Select medical data type
    if expert_role == "CKD慢性肾病专家":
        medical_df_list = list(medical_data_ckd.values())
        prompt_medical = prompt_medical_ckd
    elif expert_role == "血液透析专家":
        medical_df_list = list(medical_data_dialysis.values())
        prompt_medical = prompt_medical_dialysis
    else:
        st.warning("请选择有效的专家角色。")
        st.stop()
    with st.expander("医疗Prompt", expanded=False):
        st.write(prompt_medical)
with col2:
    # Add any other input parameters here

    st.write(f"{expert_role}: 医疗数据分析")
    for i, medical_df in enumerate(medical_df_list, start=1):
        with st.expander(f"患者医疗数据 {i}", expanded=False):
            st.write(medical_df)



# Analysis section
st.header("分析结果")
if st.button('开始分析'):
    st.markdown("## 分析结果")

    # Progress bar
    progress_bar = st.progress(0)
    total_analyses = len(prompt_medical)
    progress_step = 1 / total_analyses

    # Medical Analysis
    for i, medical_prompt in enumerate(prompt_medical, start=1):
        with st.expander(f"医疗分析 - 患者 {i}", expanded=False):
            tab1, tab2, tab3, tab4, tab5= st.tabs(["医疗数据预览", "GPT-4o", "Gemini Pro", "Claude-3", "Llama-3.1"])

            with tab1:
                st.write(medical_df_list[i-1])

            with tab2:
                st.subheader(f"GPT-4o 医疗分析 - 患者 {i}")
                model_type, insights = generate_gpt_insights(medical_prompt, expert_role)
                st.write(insights)

            with tab3:
                st.subheader(f"Gemini Pro 医疗分析 - 患者 {i}")
                insights = generate_gemini_insights(medical_prompt, expert_role)
                st.write(insights)

            with tab4:
                st.subheader(f"Claude-sonnet 医疗分析 - 患者 {i}")
                insights = generate_anthropic_insights(medical_prompt, expert_role)
                st.write(insights)

            with tab5:
                st.subheader(f"Llama-3.1 医疗分析 - 患者 {i}")
                if not test_llama_connection():
                    st.error("无法连接到Llama服务器。请确保Llama服务器正在运行。")
                    continue
                insights = generate_llama_insights(medical_prompt, expert_role)
                st.write(insights)

        progress_bar.progress(i * progress_step)

else:
    st.info("点击'开始分析'以开始。")