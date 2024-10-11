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
from files.AI_system_and_prompt import system_message_single_patient_chinese, prompt_single_patient_chinese, system_message_multiple_patient_chinese, prompt_multiple_patient_chinese

# Set page configuration
st.set_page_config(layout="centered", page_title="AI 透析患者数据分析-单一患者")

# Access API keys from secrets
openai_api_key = st.secrets["api_keys"]["openai_api_key"]
gemini_api_key = st.secrets["api_keys"]["gemini_api_key"]
anthropic_api_key = st.secrets["api_keys"]["anthropic_api_key"]

# Function definitions
@st.cache_data
def generate_anthropic_insights(prompt, expert_role, system_message_template):
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
def generate_gpt_insights(prompt, expert_role, system_message_template):
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

# Initialize session state
if 'selected_patient' not in st.session_state:
    st.session_state.selected_patient = None

def generate_structured_prompt_single_patient(lab_data, treatment_data):
    prompt = prompt_single_patient_chinese.format(
        name=lab_data['患者信息']['姓名'],
        age=lab_data['患者信息']['年龄'],
        gender=lab_data['患者信息']['性别'],
        lab_results=json.dumps(lab_data, ensure_ascii=False, indent=2),
        treatment_data=treatment_data.to_markdown(index=False)
    )
    return prompt

def generate_structured_prompt_multiple_patient(lab_data, treatment_data):
    # Calculate basic statistics
    total_patients = len(set(treatment_data['Patient']))
    patient_ages = [patient['患者信息']['年龄'] for patient in lab_data]
    gender_distribution = {patient['患者信息']['性别']: 0 for patient in lab_data}
    for patient in lab_data:
        gender_distribution[patient['患者信息']['性别']] += 1

    # Calculate descriptive statistics for key metrics
    hemoglobin_values = [patient['血常规']['血红蛋白'] for patient in lab_data]
    albumin_values = [patient['肝功能']['白蛋白'] for patient in lab_data]
    phosphorus_values = [patient['电解质']['无机磷'] for patient in lab_data]
    calcium_values = [patient['电解质']['钙'] for patient in lab_data]
    potassium_values = [patient['电解质']['钾'] for patient in lab_data]
    
    # Format the prompt
    prompt = prompt_multiple_patient_chinese.format(
        total_patients=total_patients,
        lab_results=json.dumps(lab_data, ensure_ascii=False, indent=2),
        treatment_data=treatment_data.to_markdown(index=False),
        # Add other necessary data for overview_table and key_metrics_table
    )
    return prompt

# Streamlit app
st.title("AI 透析患者数据分析")

st.markdown("""
这个应用程序利用多个AI模型（GPT-4、Gemini Pro、Claude-3）来分析透析患者的医疗数据。
它提供基于最新临床指南的详细医疗分析和个性化治疗建议，包括一个月的透析治疗数据和趋势图表。
""")

tab1, tab2 = st.tabs(["多名患者", "单一患者"])

# Convert list to dictionary (move this outside of the tabs)
all_lab_data_dict = {patient['患者信息']['姓名']: patient for patient in all_lab_data}

with tab1:
    st.header("患者数据")
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("检查结果"):
            st.json(all_lab_data)
    with col2:
        with st.expander("透析治疗数据（一个月）"):
            st.dataframe(all_treatment_data)

    st.header("AI 分析结果")
    if st.button('开始分析', key='multi_patient_analysis'):

        system_message_template = system_message_multiple_patient_chinese
        prompt = generate_structured_prompt_multiple_patient(all_lab_data, all_treatment_data)

        # Progress bar
        progress_bar = st.progress(0)
        total_analyses = 3  # GPT, Gemini, Claude
        progress_step = 1 / total_analyses

        tab1, tab2, tab3 = st.tabs(["GPT-4", "Gemini Pro", "Claude-3"])

        with tab1:
            st.subheader("GPT-4 医疗分析")
            with st.spinner('GPT-4 分析中，请稍候...'):
                start_time_gpt = time.time()
                model_type, gpt_insights = generate_gpt_insights(prompt, "透析和肾脏病专家", system_message_template)
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
                claude_insights = generate_anthropic_insights(prompt, "透析和肾脏病专家", system_message_template)
                end_time_claude = time.time()
                st.info(f"Claude-3 分析耗时: {end_time_claude - start_time_claude:.2f} 秒")
                st.markdown(claude_insights)
                progress_bar.progress(3 * progress_step)
    else:
        st.info("点击'开始分析'以开始。")

with tab2:
    # Patient selection
    st.header("患者选择")
    selected_patient = st.selectbox("选择要分析的患者", patients, index=patients.index(st.session_state.selected_patient) if st.session_state.selected_patient else 0)

    if selected_patient:
        st.session_state.selected_patient = selected_patient  # Update session state
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

        st.header("AI 分析结果")
        if st.button('开始分析', key='single_patient_analysis'):

            system_message_template = system_message_single_patient_chinese
            prompt = generate_structured_prompt_single_patient(all_lab_data_dict[selected_patient], patient_treatment_data)
            
            # Progress bar
            progress_bar = st.progress(0)
            total_analyses = 3  # GPT, Gemini, Claude
            progress_step = 1 / total_analyses

            tab1, tab2, tab3 = st.tabs(["GPT-4", "Gemini Pro", "Claude-3"])

            with tab1:
                st.subheader("GPT-4 医疗分析")
                with st.spinner('GPT-4 分析中，请稍候...'):
                    start_time_gpt = time.time()
                    model_type, gpt_insights = generate_gpt_insights(prompt, "透析和肾脏病专家", system_message_template)
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
                    claude_insights = generate_anthropic_insights(prompt, "透析和肾脏病专家", system_message_template)
                    end_time_claude = time.time()
                    st.info(f"Claude-3 分析耗时: {end_time_claude - start_time_claude:.2f} 秒")
                    st.markdown(claude_insights)
                    progress_bar.progress(3 * progress_step)
    else:
        st.info("请选择一个患者进行分析。")

st.markdown("---")
st.markdown("© 2024 AI 透析患者数据分析应用. All rights reserved.")