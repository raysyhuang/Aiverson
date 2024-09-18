import streamlit as st
import pandas as pd
import math
import sqlite3
from datetime import datetime

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
def save_to_db(age, sex, egfr, acr, serum_creatinine, hemoglobin, calcium, phosphate, bicarbonate, albumin, systolic_bp, risk_2yr, risk_5yr):
    conn = sqlite3.connect('kf_risk.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO kf_risk_results (age, sex, egfr, acr, serum_creatinine, hemoglobin, calcium, phosphate, bicarbonate, albumin, systolic_bp, risk_2yr, risk_5yr)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (age, sex, egfr, acr, serum_creatinine, hemoglobin, calcium, phosphate, bicarbonate, albumin, systolic_bp, risk_2yr, risk_5yr))
    conn.commit()
    conn.close()

# Function to retrieve data from the database
def get_results_from_db():
    conn = sqlite3.connect('kf_risk.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM kf_risk_results')
    rows = cursor.fetchall()  # Fetch all rows
    conn.close()
    return rows

# Updated coefficients for Kidney Failure Risk based on official equation
def kidney_failure_risk(age, sex, egfr, acr, serum_creatinine, hemoglobin, calcium, phosphate, systolic_bp):
    # Adjusted intercepts for 2-year and 5-year risk (based on official sources)
    intercept_2yr = 0.99  
    intercept_5yr = 1.42  
    
    # Adjusted coefficients
    coef_age = 0.046
    coef_sex = 0.21
    coef_egfr = -0.015
    coef_acr = 0.39
    coef_serum_creatinine = 0.28
    coef_hemoglobin = -0.19
    coef_calcium = -0.32
    coef_phosphate = 0.35
    coef_systolic_bp = 0.013

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

# Create the database and table if it doesn't exist
create_db()

# Streamlit App Title
st.title("肾功能衰竭风险预测工具 / Kidney Failure Risk Prediction Tool")

st.write("请输入患者的相关信息，或上传包含多个患者数据的Excel文件。 / Please enter patient information or upload an Excel file containing multiple patients' data.")

# Use columns to organize manual input fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("年龄 (Age)", min_value=1, max_value=100, value=58)
    sex = st.selectbox("性别 (Sex)", ["Male", "Female"], index=0)
    egfr = st.number_input("估算肾小球滤过率 (eGFR) (mL/min/1.73m²)", min_value=1.0, max_value=100.0, value=24.0)
    acr = st.number_input("尿白蛋白/肌酐比率 (ACR) (mg/g)", min_value=0.0, max_value=5000.0, value=530.0)

with col2:
    serum_creatinine = st.number_input("血清肌酐 (Serum Creatinine) (mg/dL)", min_value=0.1, max_value=10.0, value=2.6)
    hemoglobin = st.number_input("血红蛋白 (Hemoglobin) (g/dL)", min_value=5.0, max_value=20.0, value=10.5)
    calcium = st.number_input("血清钙 (Calcium) (mg/dL)", min_value=5.0, max_value=15.0, value=8.9)
    phosphate = st.number_input("血清磷 (Phosphate) (mg/dL)", min_value=1.0, max_value=10.0, value=4.9)

# Add additional fields for bicarbonate and albumin
col3, col4 = st.columns(2)

with col3:
    bicarbonate = st.number_input("碳酸氢盐 (Bicarbonate) (mmol/L)", min_value=0.0, max_value=50.0, value=24.0)
    albumin = st.number_input("白蛋白 (Albumin) (g/dL)", min_value=1.0, max_value=15.0, value=3.5)

with col4:
    systolic_bp = st.number_input("收缩压 (Systolic BP) (mmHg)", min_value=80, max_value=200, value=148)

# Calculate the risk when the button is clicked
if st.button("计算风险 / Calculate Risk"):
    # Call the kidney_failure_risk function with the input parameters
    risk_2yr, risk_5yr = kidney_failure_risk(age, sex, egfr, acr, serum_creatinine, hemoglobin, calcium, phosphate, systolic_bp)
    
    # Determine the CKD stage based on eGFR
    ckd_stage = determine_ckd_stage(egfr)
    
    # Display risk results and CKD stage
    st.subheader("评估结果 / Assessment")
    st.write(f"**eGFR:** {egfr} mL/min/1.73m² - **{ckd_stage}**")
    st.write(f"**2年内肾衰竭风险:** {risk_2yr}% / **2-Year Kidney Failure Risk:** {risk_2yr}%")
    st.write(f"**5年内肾衰竭风险:** {risk_5yr}% / **5-Year Kidney Failure Risk:** {risk_5yr}%")
    
    # Add explanatory text for CKD stages and lab results
    st.write("Glomerular Filtration Rate (GFR) is the best test to tell you how well your kidneys are cleaning your blood. GFR is calculated using your blood creatinine test, age, and sex. A lower GFR indicates worse kidney function. Below 30, you should see a nephrologist.")
    st.write("Albumin, Phosphate, and Corrected Calcium levels provide additional insights into your kidney function and overall health.")
    
    # Save the results to the SQLite database
    save_to_db(age, sex, egfr, acr, serum_creatinine, hemoglobin, calcium, phosphate, bicarbonate, albumin, systolic_bp, risk_2yr, risk_5yr)

# Button to show saved results
if st.button("查看保存的结果 / View Saved Results"):
    results = get_results_from_db()
    if results:
        st.write("保存的肾功能衰竭风险结果 / Saved Kidney Failure Risk Results:")
        st.write(pd.DataFrame(results, columns=['ID', 'Age', 'Sex', 'eGFR', 'ACR', 'Serum Creatinine', 'Hemoglobin', 'Calcium', 'Phosphate', 'Bicarbonate', 'Albumin', 'Systolic BP', '2-Year Risk', '5-Year Risk', 'Timestamp']))
    else:
        st.write("数据库中没有结果 / No results found in the database.")

# Bulk Upload Section
st.header("批量患者输入 / Bulk Patient Input")

uploaded_file = st.file_uploader("选择一个Excel文件 / Select an Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("上传的文件内容 / Uploaded file content:")
    st.write(df)
    
    if st.button("计算批量患者的风险 / Calculate Risk for Bulk Patients"):
        results = []
        for _, row in df.iterrows():
            age = row['Age']
            sex = row['Sex']
            egfr = row['eGFR']
            acr = row['ACR']
            serum_creatinine = row['Serum Creatinine']
            hemoglobin = row['Hemoglobin']
            calcium = row['Calcium']
            phosphate = row['Phosphate']
            bicarbonate = row['Bicarbonate']
            albumin = row['Albumin']
            systolic_bp = row['Systolic BP']
            
            risk_2yr, risk_5yr = kidney_failure_risk(age, sex, egfr, acr, serum_creatinine, hemoglobin, calcium, phosphate, systolic_bp)
            results.append([age, sex, egfr, acr, serum_creatinine, hemoglobin, calcium, phosphate, bicarbonate, albumin, systolic_bp, risk_2yr, risk_5yr])
        
        results_df = pd.DataFrame(results, columns=['Age', 'Sex', 'eGFR', 'ACR', 'Serum Creatinine', 'Hemoglobin', 'Calcium', 'Phosphate', 'Bicarbonate', 'Albumin', 'Systolic BP', '2-Year Risk (%)', '5-Year Risk (%)'])
        st.write("批量计算结果 / Bulk Calculation Results:")
        st.dataframe(results_df)
        
        # Download link for results
        results_df.to_excel("kf_risk_results.xlsx", index=False)
        st.download_button("下载结果 / Download Results", "kf_risk_results.xlsx")