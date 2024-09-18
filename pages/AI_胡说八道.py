from openai import OpenAI
import streamlit as st
import pandas as pd
import google.generativeai as genai


def read_excel():
    # 读取 Excel 文件中的数据
    df_read = pd.read_csv('files/data.csv')
    df_read.set_index('月份', inplace=True)
    df_read['营业外收支'] = df_read['营业外收入'] - df_read['营业外支出']
    df = df_read
    return df

df = read_excel()

def generate_gpt_insights_financial(prompt):
    # Initialize the OpenAI client with the api_key
    client = OpenAI(api_key='sk-O8I2i4KOk_ygZY7AmAQIQgXtqUZ_5tLXBd_EDuF1xRT3BlbkFJV8gG03ZH0OBZLeSSMqRxyUbjq3CkC7rgEmRYEq3rEA')
    model_type = "gpt-4"

    completion = client.chat.completions.create(
        model=model_type,
        messages=[
            {"role": "system", 
            "content": "你是一名经验丰富的财务分析师，擅长分析公司财务数据并撰写专业报告。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,
        max_tokens=1500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return model_type, completion.choices[0].message.content

def generate_gpt_insights_medical(prompt):
    # Initialize the OpenAI client with the api_key
    client = OpenAI(api_key='sk-O8I2i4KOk_ygZY7AmAQIQgXtqUZ_5tLXBd_EDuF1xRT3BlbkFJV8gG03ZH0OBZLeSSMqRxyUbjq3CkC7rgEmRYEq3rEA')
    model_type = "gpt-4o"

    completion = client.chat.completions.create(
        model=model_type,
        messages=[
            {
                "role": "system", 
                "content": "你是一名经验丰富的肾病专家，擅长慢性肾病（CKD）、透析及相关治疗。你将根据患者数据，使用肾衰竭风险方程（KFRE）等模型进行全面评估，并提供基于最新临床指南的治疗建议。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,  # Reduce randomness for consistent output
        max_tokens=2000,  # Ensure enough tokens for long responses
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return model_type, completion.choices[0].message.content


# Set up the API key
genai.configure(api_key="AIzaSyD97bXoPlXXEWXURaCOxL2yZWXJKhmEce0")

# Ensure the get_response function is defined
def get_response(messages, model="gemini-pro"):
    model = genai.GenerativeModel("gemini-pro")
    res = model.generate_content(messages, stream=True, safety_settings={'HARASSMENT':'block_none'})
    return res

# Function to generate insights using generative AI
def generate_gemini_insights(prompt):
    # Set up the API key
    genai.configure(api_key="AIzaSyD97bXoPlXXEWXURaCOxL2yZWXJKhmEce0")
    # Prepare the data summary
    
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    
    insights = ""
    if response.parts:
        for part in response.parts:
            if hasattr(part, 'text'):
                insights += part.text
    
    if not insights:
        insights = "目前无法生成详细的分析报告。"
    
    return insights


prompt = f"""
请对以下公司的财务数据进行全面分析，并将结果以 Markdown 格式输出。具体要求如下：

1. **财务数据统计摘要分析**：
- 对数据中的关键财务指标进行统计分析，包括趋势、同比和环比变化等。
- 提供对这些指标的解释和洞察。

2. **公司运营状况分析**：
- 从盈利能力、运营效率、流动性和偿债能力等方面评估公司的运营状况。
- 指出可能的风险和机会，并提供相应的建议。

3. **营业收入和毛利表格**：
- 重新整理并输出包含月份、年度、营业收入和毛利的表格。
- 确保表格格式清晰，数据准确，便于阅读和理解。

**数据如下**：

{df.to_markdown()}
"""

medical_df1 = f"""
## 患者信息：
### 肾衰竭风险方程参数：
1. 年龄: 78岁
2. 性别: 男性
3. eGFR: 15 mL/min/1.73 m²
4. 血清肌酐: 5.2 mg/dL
5. 尿白蛋白/肌酐比率（ACR）: 1200 mg/g
6. 血清钙: 8.3 mg/dL
7. 血清磷: 5.5 mg/dL
8. 血红蛋白（Hb）: 9.5 g/dL
9. 收缩压: 160 mmHg
"""

medical_df2 = f"""
## 患者信息：
### 肾衰竭风险方程参数：
1. 年龄：58岁
2. 性别：男性
3. eGFR：24 mL/min/1.73 m²
4. 血清肌酐：2.6 mg/dL
5. 尿白蛋白/肌酐比率（ACR）：530 mg/g
6. 血清钙：8.9 mg/dL
7. 血清磷：4.9 mg/dL
8. 血红蛋白（Hb）：10.5 g/dL
9. 收缩压：148 mmHg
"""

medical_df3 = f"""
## 患者信息：
### 肾衰竭风险方程参数：
1. 年龄: 62岁
2. 性别: 男性
3. eGFR: 8 mL/min/1.73 m²
4. 血清肌酐: 6.5 mg/dL 
5. 尿白蛋白/肌酐比率（ACR）: 1500 mg/g
6. 血清钙: 7.8 mg/dL
7. 血清磷: 6.2 mg/dL
8. 血红蛋白（Hb）: 8.2 g/dL
9. 收缩压: 180 mmHg
10. 症状: 恶心、呕吐、乏力、呼吸困难，伴有水肿
"""


# Example default values or values retrieved from data
ESRD_risk_percentage = 1  # Placeholder percentage for ESRD risk
eGFR_value = 1  # Example value for eGFR in mL/min/1.73 m²
ACR_value = 1  # Example value for ACR in mg/g
age = 1  # Example patient age
gender = "男性"  # Example gender
systolic_bp = 1  # Example systolic blood pressure
hb_value = 1  # Example hemoglobin level



medical_df_list = [medical_df1, medical_df2, medical_df3]  # Define the list

prompt_medical = []
for medical_df in medical_df_list:
    prompt_medical.append(f"""
    数据如下：
    {medical_df}

    请根据以下格式撰写患者的医疗评估报告，并根据患者的具体值提供动态的分析和管理建议：

    ## 患者评估报告

    ## 1. 风险评估
    - ESRD风险: 根据肾衰竭风险方程 (KFRE) 计算，患者在两年内进展至需要透析或肾移植的终末期肾病 (ESRD) 的风险约为 **{ESRD_risk_percentage}%**。
    - eGFR: {eGFR_value} mL/min/1.73 m²
    - ACR: {ACR_value} mg/g
    - 分析: 患者的 eGFR 和 ACR 值显著增加了进展至 ESRD 的风险。eGFR 越低，ACR 越高，意味着肾功能损伤更严重，未来 ESRD 的可能性更大。根据这些值，患者在未来两年内的 ESRD 风险为 **{ESRD_risk_percentage}%**。

    ### 2. CKD 与其他因素的关系
    - CKD与年龄: 患者年龄为 {age} 岁，老龄化会加剧 CKD 的进展。年龄越大，肾脏的恢复能力越差，病情恶化的速度可能更快。
    - CKD与性别: 患者为 {gender}，性别可能对 CKD 的进展有一定影响。通常男性比女性进展至 ESRD 的风险更高。

    ### 3. 分期和进展时间
    - 患者的 CKD 分期为 **CKD 第X期** (eGFR: {eGFR_value} mL/min/1.73 m²)。
    - 分析: 根据患者的 eGFR 值，当前处于 CKD 的第X期，预计患者可能在未来 **X年** 内需要透析。eGFR 的下降速度决定了具体透析时间，需定期监测。

    ### 4. 临床问题分析
    - 高血压: 收缩压为 {systolic_bp} mmHg，尽管可能已使用双重降压治疗，仍未得到充分控制。血压控制不佳将加速肾功能的进一步恶化。
    - 贫血: 血红蛋白为 {hb_value} g/dL，提示慢性疾病性贫血。贫血是 CKD 患者常见的并发症，需进一步管理。
    - 蛋白尿: ACR 值为 {ACR_value} mg/g，提示严重的蛋白尿。蛋白尿是肾损伤的标志，需针对蛋白尿进行治疗。

    ### 5. 治疗建议
    - 血压管理: 
    - 糖尿病控制: 
    - 贫血管理: 
    - 蛋白尿管理: 
    - 继发性甲状旁腺功能亢进: 

    ### 6. 随访建议
    - 定期随访：
    - 营养咨询：

    ### 7. 患者教育
    - 饮食: 遵循低钾、低磷、低钠饮食，限制水分摄入，避免高蛋白食品，以减轻透析负担？
    - 运动: 建议进行适度运动，如散步，有助于维持体力和改善心血管功能？
    - 透析管理与计划: 向患者详细解释透析的过程、可能的并发症以及透析后应注意的事项，帮助患者适应透析生活？

    ### 8. 简要分析：
    - CKD阶段: 患者的eGFR为{eGFR_value} mL/min/1.73 m²，已经处于 CKD第5期（终末期肾病, ESRD），这是肾功能的最终阶段，肾脏无法维持正常功能，透析已是迫在眉睫的治疗选择？
	- ACR水平: {ACR_value} mg/g，表明肾脏损伤极为严重，蛋白尿程度极高？
	- 血压控制: 收缩压高达{systolic_bp} mmHg，表明高血压控制不良，且对肾脏和心血管系统的损伤已经非常显著？
	- 贫血: 血红蛋白仅为8.2 g/dL，提示严重贫血，可能加剧患者的疲劳感和心脏负担？
	- 钙磷代谢紊乱: 血清钙偏低，血清磷升高，提示继发性甲状旁腺功能亢进，钙磷代谢严重紊乱，可能导致骨病或血管钙化等并发症？
    - 症状表现: 患者已表现出严重的尿毒症症状，如恶心、呕吐、乏力、呼吸困难和水肿，这些都是提示立即开始透析的紧急信号？
    """
    )

st.title("AI 数据分析 GPT & Gemini")
st.markdown("---")
st.write("财务数据分析")
with st.expander("财务数据预览", expanded=False):
    st.write(df)
with st.expander("财务Prompt", expanded=False):
    st.write(prompt)

st.write("医疗数据分析")
with st.expander("患者医疗数据 1", expanded=False):
    st.write(medical_df1)
with st.expander("患者医疗数据 2", expanded=False):
    st.write(medical_df2)
with st.expander("患者医疗数据 3", expanded=False):
    st.write(medical_df3)
with st.expander("医疗Prompt", expanded=False):
    st.write(prompt_medical)

tab1, tab2 = st.tabs(["GPT-4o", "Gemini Pro"])

with tab1:
    st.subheader("GPT-4o")
    st.write("财务数据分析")

    if st.button("生成详细分析(财务)", key="gpt_button_financial"):
        model_type, insights = generate_gpt_insights_financial(prompt)
        st.write(insights)

    st.write("医疗数据分析")
    if st.button("生成详细分析(患者 1)", key="gpt_button_medical1"):
        model_type, insights = generate_gpt_insights_medical(prompt_medical[0])
        st.write(medical_df1)
        st.write(insights)

    if st.button("生成详细分析(患者 2)", key="gpt_button_medical2"):
        model_type, insights = generate_gpt_insights_medical(prompt_medical[1])   
        st.write(medical_df2)
        st.write(insights)

    if st.button("生成详细分析(患者 3)", key="gpt_button_medical3"):
        model_type, insights = generate_gpt_insights_medical(prompt_medical[2])
        st.write(medical_df3)
        st.write(insights)

with tab2:
    st.subheader("Gemini Pro")
    st.write("财务数据分析")
    if st.button("生成详细分析(财务)", key="gemini_button_financial"):
        insights = generate_gemini_insights(prompt)
        st.write(insights)

    st.write("医疗数据分析")
    if st.button("生成详细分析(患者 1)", key="gemini_button_medical1"):
        insights = generate_gemini_insights(prompt_medical[0])
        st.write(medical_df1)
        st.write(insights)

    if st.button("生成详细分析(患者 2)", key="gemini_button_medical2"):
        insights = generate_gemini_insights(prompt_medical[1])
        st.write(medical_df2)
        st.write(insights)

    if st.button("生成详细分析(患者 3)", key="gemini_button_medical3"):
        insights = generate_gemini_insights(prompt_medical[2])
        st.write(medical_df3)
        st.write(insights)



