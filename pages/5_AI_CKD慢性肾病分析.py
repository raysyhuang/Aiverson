import streamlit as st
import pandas as pd
import math
import sqlite3
import os
import logging
from datetime import datetime
from openai import OpenAI
import google.generativeai as genai
import anthropic
from files.medical_data import medical_data_ckd, medical_data_dialysis, dialysis_prompt, CKD_prompt, system_message_template  # Import the data and prompts


# Access API keys from secrets
openai_api_key = st.secrets["api_keys"]["openai_api_key"]
gemini_api_key = st.secrets["api_keys"]["gemini_api_key"]
anthropic_api_key = st.secrets["api_keys"]["anthropic_api_key"]

# Configure logging
logging.basicConfig(level=logging.INFO)

# Function to create the SQLite database and table if not exists
def create_db():
    conn = sqlite3.connect('kf_risk.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS kf_risk_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age INTEGER,
            sex TEXT,
            egfr REAL,
            acr REAL,
            serum_creatinine REAL,
            hemoglobin REAL,
            calcium REAL,
            phosphate REAL,
            bicarbonate REAL,
            albumin REAL,
            systolic_bp REAL,
            risk_2yr REAL,
            risk_5yr REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Function to save inputs and results into the database
def save_to_db(age, sex, egfr, acr, serum_creatinine, hemoglobin, calcium,
               phosphate, bicarbonate, albumin, systolic_bp, risk_2yr, risk_5yr):
    try:
        conn = sqlite3.connect('kf_risk.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO kf_risk_results (age, sex, egfr, acr, serum_creatinine, hemoglobin,
                                         calcium, phosphate, bicarbonate, albumin, systolic_bp,
                                         risk_2yr, risk_5yr)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (age, sex, egfr, acr, serum_creatinine, hemoglobin, calcium, phosphate,
              bicarbonate, albumin, systolic_bp, risk_2yr, risk_5yr))
        conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
    finally:
        conn.close()

# Function to retrieve data from the database
def get_results_from_db():
    try:
        conn = sqlite3.connect('kf_risk.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM kf_risk_results')
        rows = cursor.fetchall()
        return rows
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        return []
    finally:
        conn.close()

# Function to calculate kidney failure risk with adjusted coefficients
def kidney_failure_risk(age, sex, egfr, acr, serum_creatinine=0, hemoglobin=0,
                        calcium=0, phosphate=0, systolic_bp=0):
    # Adjusted intercepts and coefficients based on updated research
    intercept_2yr = -0.5  # Placeholder value
    intercept_5yr = 0.0   # Placeholder value

    coef_age = 0.05       # Placeholder value
    coef_sex = 0.2        # Placeholder value (1 for male, 0 for female)
    coef_egfr = -0.016    # Placeholder value
    coef_acr = 0.4        # Placeholder value (log-transformed)
    coef_serum_creatinine = 0.3    # Placeholder value
    coef_hemoglobin = -0.18        # Placeholder value
    coef_calcium = -0.3            # Placeholder value
    coef_phosphate = 0.36          # Placeholder value
    coef_systolic_bp = 0.014       # Placeholder value

    # Log-transform ACR (Albumin-Creatinine Ratio)
    log_acr = math.log(acr + 1)

    # Calculate log risk for 2-year and 5-year progression
    log_risk_2yr = (intercept_2yr +
                    coef_age * age +
                    coef_sex * (1 if sex.lower() == 'male' else 0) +
                    coef_egfr * egfr +
                    coef_acr * log_acr +
                    coef_serum_creatinine * serum_creatinine +
                    coef_hemoglobin * hemoglobin +
                    coef_calcium * calcium +
                    coef_phosphate * phosphate +
                    coef_systolic_bp * systolic_bp)

    log_risk_5yr = (intercept_5yr +
                    coef_age * age +
                    coef_sex * (1 if sex.lower() == 'male' else 0) +
                    coef_egfr * egfr +
                    coef_acr * log_acr +
                    coef_serum_creatinine * serum_creatinine +
                    coef_hemoglobin * hemoglobin +
                    coef_calcium * calcium +
                    coef_phosphate * phosphate +
                    coef_systolic_bp * systolic_bp)

    # Convert log risk to percentage risk using logistic function
    risk_2yr = 1 / (1 + math.exp(-log_risk_2yr)) * 100
    risk_5yr = 1 / (1 + math.exp(-log_risk_5yr)) * 100

    return round(risk_2yr, 2), round(risk_5yr, 2)

# Function to determine CKD stage based on eGFR
def determine_ckd_stage(egfr):
    if egfr >= 90:
        return "Stage 1: Normal Kidney Function"
    elif 60 <= egfr < 90:
        return "Stage 2: Mild Decrease in Function"
    elif 30 <= egfr < 60:
        return "Stage 3: Moderate Decrease in Function"
    elif 15 <= egfr < 30:
        return "Stage 4: Severe Decrease in Function"
    else:
        return "Stage 5: Kidney Failure"

# Function definitions for AI analysis
def generate_gpt_insights(prompt, expert_role):
    try:
        client = OpenAI(api_key=openai_api_key)
        system_message = system_message_template.format(expert_role=expert_role)
        completion = client.chat.completions.create(
            model="gpt-4o",
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
        return completion.choices[0].message.content
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        return f"OpenAI API 出错: {str(e)}. 跳过此分析。"

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
        logging.error(f"Gemini API error: {e}")
        return f"Gemini API 出错: {str(e)}. 跳过此分析。"

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
        if isinstance(message.content, list) and len(message.content) > 0:
            text_block = message.content[0]
            if hasattr(text_block, 'text'):
                return text_block.text
            elif isinstance(text_block, str):
                return text_block.split("text='", 1)[1].split("', type='text')", 1)[0]
        return "无法提取文本内容"
    except Exception as e:
        logging.error(f"Anthropic API error: {e}")
        return f"Anthropic API 出错: {str(e)}. 跳过此分析。"

# Create the database and table if it doesn't exist
create_db()

# Streamlit App Title
st.set_page_config(layout="centered", page_title="AI 数据分析和肾功能衰竭风险预测工具")
st.title("AI 数据分析和肾功能衰竭风险预测工具")
st.markdown("""
这个应用程序结合了肾功能衰竭风险预测和AI模型分析。
您可以选择分析预定义的医疗数据，或者输入患者的相关信息，或上传包含多个患者数据的Excel文件。
应用程序将计算肾衰竭风险，并可选择使用多个AI模型（GPT-4、Gemini Pro和Claude-3）来分析患者的医疗数据。
""")

# Input Section
st.header("选择分析方式")
analysis_method = st.radio("选择分析方式", ("预定义医疗数据分析", "手动输入", "上传Excel文件"))

if analysis_method == "预定义医疗数据分析":
    # Load predefined medical data
    with st.expander("预定义医疗数据", expanded=False):
        medical_data_ckd, medical_data_dialysis, dialysis_prompt, CKD_prompt
    if not medical_data_ckd or not medical_data_dialysis:
        st.stop()

    expert_role = st.selectbox("选择专家角色", ["CKD慢性肾病专家", "血液透析专家"])
    # Select medical data type
    if expert_role == "CKD慢性肾病专家":
        medical_df_list = list(medical_data_ckd.values())
        prompt_medical = []
        for medical_df in medical_data_ckd.values():
            # Extract necessary values for KFRE calculation
            # Parsing the medical_df string to extract values
            lines = medical_df.strip().split('\n')
            data = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip(' -.')
                    value = value.strip()
                    data[key] = value

            # Extract parameters
            age = int(data.get('年龄', '0').replace('岁', '').strip())
            sex = data.get('性别', '').strip()
            egfr = float(data.get('eGFR', '0').split()[0])
            acr = float(data.get('尿白蛋白/肌酐比率（ACR）', '0').split()[0])
            serum_creatinine = float(data.get('血清肌酐', '0').split()[0])
            hemoglobin = float(data.get('血红蛋白（Hb）', '0').split()[0])
            calcium = float(data.get('血清钙', '0').split()[0])
            phosphate = float(data.get('血清磷', '0').split()[0])
            systolic_bp = float(data.get('收缩压', '0').split()[0])

            # Calculate KFRE risk
            risk_2yr, risk_5yr = kidney_failure_risk(
                age, sex, egfr, acr,
                serum_creatinine or 0,
                hemoglobin or 0,
                calcium or 0,
                phosphate or 0,
                systolic_bp or 0
            )

            ESRD_risk_percentage = risk_2yr  # Using 2-year risk as example

            # Prepare prompt
            prompt = CKD_prompt.format(
                medical_df=medical_df,
                ESRD_risk_percentage=ESRD_risk_percentage,
                eGFR_value=egfr,
                ACR_value=acr,
                systolic_bp=systolic_bp,
                hb_value=hemoglobin
            )
            prompt_medical.append(prompt)
    elif expert_role == "血液透析专家":
        medical_df_list = list(medical_data_dialysis.values())
        prompt_medical = []
        for medical_df in medical_data_dialysis.values():
            prompt = dialysis_prompt.format(
                medical_df=medical_df
            )
            prompt_medical.append(prompt)
    else:
        st.warning("请选择有效的专家角色。")
        st.stop()

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
                tab1, tab2, tab3, tab4 = st.tabs(["医疗数据预览", "GPT-4", "Gemini Pro", "Claude-3"])

                with tab1:
                    st.write(medical_df_list[i-1])

                with tab2:
                    st.subheader(f"GPT-4 医疗分析 - 患者 {i}")
                    insights = generate_gpt_insights(medical_prompt, expert_role)
                    st.write(insights)

                with tab3:
                    st.subheader(f"Gemini Pro 医疗分析 - 患者 {i}")
                    insights = generate_gemini_insights(medical_prompt, expert_role)
                    st.write(insights)

                with tab4:
                    st.subheader(f"Claude-3 医疗分析 - 患者 {i}")
                    insights = generate_anthropic_insights(medical_prompt, expert_role)
                    st.write(insights)

            progress_bar.progress(i * progress_step)

    else:
        st.info("点击'开始分析'以开始。")

elif analysis_method == "手动输入":
    st.header("输入参数")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("年龄 (Age)", min_value=1, max_value=100, value=58)
        sex = st.selectbox("性别 (Sex)", ["Male", "Female"], index=0)
        egfr = st.number_input("估算肾小球滤过率 (eGFR) (mL/min/1.73m²)", min_value=1.0, max_value=150.0, value=24.0)
        acr = st.number_input("尿白蛋白/肌酐比率 (ACR) (mg/g)", min_value=0.0, max_value=5000.0, value=530.0)

    with col2:
        serum_creatinine = st.number_input("血清肌酐 (Serum Creatinine) (mg/dL)", min_value=0.0, max_value=10.0, value=2.6)
        hemoglobin = st.number_input("血红蛋白 (Hemoglobin) (g/dL)", min_value=0.0, max_value=20.0, value=10.5)
        calcium = st.number_input("血清钙 (Calcium) (mg/dL)", min_value=0.0, max_value=15.0, value=8.9)
        phosphate = st.number_input("血清磷 (Phosphate) (mg/dL)", min_value=0.0, max_value=10.0, value=4.9)

    col3, col4 = st.columns(2)

    with col3:
        bicarbonate = st.number_input("碳酸氢盐 (Bicarbonate) (mmol/L)", min_value=0.0, max_value=50.0, value=24.0)
        albumin = st.number_input("白蛋白 (Albumin) (g/dL)", min_value=0.0, max_value=15.0, value=3.5)

    with col4:
        systolic_bp = st.number_input("收缩压 (Systolic BP) (mmHg)", min_value=0, max_value=300, value=148)

    if st.button("计算风险并进行AI分析"):
        risk_2yr, risk_5yr = kidney_failure_risk(
            age, sex, egfr, acr,
            serum_creatinine or 0,
            hemoglobin or 0,
            calcium or 0,
            phosphate or 0,
            systolic_bp or 0
        )
        ckd_stage = determine_ckd_stage(egfr)

        st.subheader("评估结果 / Assessment")
        st.write(f"**eGFR:** {egfr} mL/min/1.73m² - **{ckd_stage}**")
        st.write(f"**2年内肾衰竭风险:** {risk_2yr}%")
        st.write(f"**5年内肾衰竭风险:** {risk_5yr}%")

        save_to_db(
            age, sex, egfr, acr,
            serum_creatinine or 0,
            hemoglobin or 0,
            calcium or 0,
            phosphate or 0,
            bicarbonate or 0,
            albumin or 0,
            systolic_bp or 0,
            risk_2yr, risk_5yr
        )

        medical_data = {
            'Age': age,
            'Sex': sex,
            'eGFR': egfr,
            'ACR': acr,
            'Serum Creatinine': serum_creatinine,
            'Hemoglobin': hemoglobin,
            'Calcium': calcium,
            'Phosphate': phosphate,
            'Bicarbonate': bicarbonate,
            'Albumin': albumin,
            'Systolic BP': systolic_bp
            # '2-Year Risk (%)': risk_2yr,
            # '5-Year Risk (%)': risk_5yr
        }

        medical_df = pd.DataFrame([medical_data])

        st.subheader("AI 分析")
        expert_role = "CKD慢性肾病专家"
        prompt = f"请根据以下患者数据进行分析：\n{medical_df.to_string(index=False)}\n基于最新临床指南，提供治疗建议。"
        with st.expander("查看患者数据"):
            st.write(prompt) 
            st.write(medical_df)

        tab1, tab2, tab3 = st.tabs(["GPT-4", "Gemini Pro", "Claude-3"])

        with tab1:
            st.write("GPT-4 分析结果：")
            insights = generate_gpt_insights(prompt, expert_role)
            st.write(insights)

        with tab2:
            st.write("Gemini Pro 分析结果：")
            insights = generate_gemini_insights(prompt, expert_role)
            st.write(insights)

        with tab3:
            st.write("Claude-3 分析结果：")
            insights = generate_anthropic_insights(prompt, expert_role)
            st.write(insights)

elif analysis_method == "上传Excel文件":
    uploaded_file = st.file_uploader("选择一个Excel文件 / Select an Excel file", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.write("上传的文件内容 / Uploaded file content:")
        st.write(df)

        if st.button("计算批量患者的风险并进行AI分析"):
            results = []
            for index, row in df.iterrows():
                age = row['Age']
                sex = row['Sex']
                egfr = row['eGFR']
                acr = row['ACR']
                serum_creatinine = row.get('Serum Creatinine', 0)
                hemoglobin = row.get('Hemoglobin', 0)
                calcium = row.get('Calcium', 0)
                phosphate = row.get('Phosphate', 0)
                bicarbonate = row.get('Bicarbonate', 0)
                albumin = row.get('Albumin', 0)
                systolic_bp = row.get('Systolic BP', 0)

                risk_2yr, risk_5yr = kidney_failure_risk(
                    age, sex, egfr, acr,
                    serum_creatinine or 0,
                    hemoglobin or 0,
                    calcium or 0,
                    phosphate or 0,
                    systolic_bp or 0
                )
                results.append([
                    age, sex, egfr, acr, serum_creatinine, hemoglobin,
                    calcium, phosphate, bicarbonate, albumin, systolic_bp,
                    risk_2yr, risk_5yr
                ])

                save_to_db(
                    age, sex, egfr, acr,
                    serum_creatinine or 0,
                    hemoglobin or 0,
                    calcium or 0,
                    phosphate or 0,
                    bicarbonate or 0,
                    albumin or 0,
                    systolic_bp or 0,
                    risk_2yr, risk_5yr
                )

                medical_data = {
                    'Age': age,
                    'Sex': sex,
                    'eGFR': egfr,
                    'ACR': acr,
                    'Serum Creatinine': serum_creatinine,
                    'Hemoglobin': hemoglobin,
                    'Calcium': calcium,
                    'Phosphate': phosphate,
                    'Bicarbonate': bicarbonate,
                    'Albumin': albumin,
                    'Systolic BP': systolic_bp,
                    '2-Year Risk (%)': risk_2yr,
                    '5-Year Risk (%)': risk_5yr
                }

                medical_df = pd.DataFrame([medical_data])

                st.subheader(f"患者 {index+1} 的AI分析")
                expert_role = st.selectbox(
                    f"选择患者 {index+1} 的专家角色",
                    ["CKD慢性肾病专家", "血液透析专家"],
                    key=f"expert_role_{index}"
                )
                prompt = f"请根据以下患者数据进行分析：\n{medical_df.to_string(index=False)}\n基于最新临床指南，提供治疗建议。"

                with st.expander(f"查看患者 {index+1} 数据"):
                    st.write(medical_df)

                tab1, tab2, tab3 = st.tabs(["GPT-4", "Gemini Pro", "Claude-3"])

                with tab1:
                    st.write("GPT-4 分析结果：")
                    insights = generate_gpt_insights(prompt, expert_role)
                    st.write(insights)

                with tab2:
                    st.write("Gemini Pro 分析结果：")
                    insights = generate_gemini_insights(prompt, expert_role)
                    st.write(insights)

                with tab3:
                    st.write("Claude-3 分析结果：")
                    insights = generate_anthropic_insights(prompt, expert_role)
                    st.write(insights)

            results_df = pd.DataFrame(
                results,
                columns=[
                    'Age', 'Sex', 'eGFR', 'ACR', 'Serum Creatinine',
                    'Hemoglobin', 'Calcium', 'Phosphate', 'Bicarbonate',
                    'Albumin', 'Systolic BP', '2-Year Risk (%)', '5-Year Risk (%)'
                ]
            )
            st.write("批量计算结果 / Bulk Calculation Results:")
            st.dataframe(results_df)

            # Provide download link
            def convert_df(df):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_excel(index=False)

            excel_data = convert_df(results_df)
            st.download_button(
                "下载结果 / Download Results",
                data=excel_data,
                file_name="kf_risk_results.xlsx"
            )

if st.button("查看保存的结果 / View Saved Results"):
    results = get_results_from_db()
    if results:
        st.write("保存的肾功能衰竭风险结果 / Saved Kidney Failure Risk Results:")
        st.write(pd.DataFrame(
            results,
            columns=[
                'ID', 'Age', 'Sex', 'eGFR', 'ACR', 'Serum Creatinine',
                'Hemoglobin', 'Calcium', 'Phosphate', 'Bicarbonate',
                'Albumin', 'Systolic BP', '2-Year Risk', '5-Year Risk',
                'Timestamp'
            ]
        ))
    else:
        st.write("数据库中没有结果 / No results found in the database.")





