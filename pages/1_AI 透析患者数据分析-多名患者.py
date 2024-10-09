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
        return f"Anthropic API error: {str(e)}. Skipping this analysis."

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

def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


# Load and preprocess data
lab_data = load_json_data('files/lab_month_50.json')
treatment_data = load_csv_data('files/treatment_month_50.csv')
#treatment_data = load_json_data('files/treatment_month_50.json')


# System message template
system_message_template = """
You are an experienced AI assistant specializing in dialysis center analysis and nephrology. Your task is to comprehensively analyze the data from multiple patients in a dialysis center over a period of about a month. You should provide detailed medical insights and recommendations for improving overall center performance and patient outcomes. Your analysis and recommendations should:

1. Be based on the latest clinical guidelines and research, such as KDIGO guidelines and relevant Chinese guidelines.
2. Focus on center-wide trends and patterns, identifying areas of strength and opportunities for improvement.
3. Analyze all key data points, including normal and abnormal values, with special attention to trends and changes over the month.
4. Explain the clinical significance of your findings and compare the center's data with standard target values.
5. Provide specific recommendations for improvement, including potential changes to center protocols, staff training, or resource allocation.
6. Reference relevant medical literature or guidelines to support your recommendations, including at least one Chinese source and international guidelines (such as KDIGO).

Please structure your analysis and recommendations as follows, using Markdown format for tables and charts where appropriate:

1. Center Overview Summary
* Brief overview of the center's key performance indicators and overall patient outcomes.
* Highlight any significant trends or changes observed over the month.

2. Key Findings and Areas of Focus
* Use a Markdown table to list 3-5 most important findings or issues at the center level.
* Each finding should include data points, clinical significance, and reasons for concern or praise.

3. Detailed Analysis
* For each aspect (see list below), please:
  * Identify abnormal or concerning data points, including normal reference ranges.
  * Explain the clinical significance of these findings, potential causes, and impacts on patient outcomes.
  * Analyze trends and changes over the month, using charts or tables to illustrate (if applicable).
  * Compare the center's data with standard guidelines or target values, citing specific guidelines and target ranges.
  * Provide specific recommendations for improvement, including actionable steps and expected outcomes.
* Aspects to analyze: 
  a. Anemia management 
  b. Mineral and bone disorder management
  c. Nutritional status
  d. Dialysis adequacy
  e. Blood pressure control and hemodynamics
  f. Fluid management
  g. Complication prevention and management
  h. Long-term outcomes and quality of life

4. Treatment Recommendations and Protocol Adjustments
* Use a Markdown table to list specific recommendations for improving center-wide protocols or practices.
* Include potential medication adjustments, dialysis prescription modifications, and center-wide policy changes.

5. Staff Training and Resource Allocation Suggestions
* Use a Markdown table to list areas where staff training or resource reallocation could improve outcomes.
* Each item should include the target area, proposed intervention, and expected benefits.

6. References and Guidelines
* List 3-4 key references or guidelines that support your recommendations.
* Include at least one Chinese source and one KDIGO guideline, noting specific chapters or page numbers.

Ensure that your response is:
* Specific and clear, directly referencing center data and trends to support your conclusions.
* Comprehensive, addressing all key aspects of dialysis center management and patient care.
* Actionable, providing clear steps that center management can take to improve outcomes.
* Balanced, acknowledging both areas of good performance and areas needing improvement.
* Evidence-based, citing relevant guidelines and research to support your recommendations.
"""

def generate_structured_prompt(lab_data, treatment_data):
    prompt = f"""
Please analyze the following data from a dialysis center, covering multiple patients over a period of about a month. Provide a comprehensive analysis of the center's performance and recommendations for improvement.

lab_data:
{json.dumps(lab_data, ensure_ascii=False, indent=2)}

treatment_data:
{treatment_data.to_markdown(index=False)}

Center Overview:
* Total Patients: 
* Average Age:  years
* Gender Distribution: Male , Female 

Key Metrics (Center Averages):
* Hemoglobin:  g/dL
* Albumin:  g/dL
* Phosphorus:    mmol/L
* Calcium:  mmol/L
* Potassium:  mmol/L

Please provide your analysis and recommendations based on this data, following the structure outlined in the system message. Focus on center-wide trends, performance indicators, and areas for improvement.

"""
    return prompt




# Streamlit app
st.set_page_config(layout="centered", page_title="AI Dialysis Center Analysis")
st.title("AI Dialysis Center Analysis")

st.markdown("""
This application uses multiple AI models (GPT-4, Gemini Pro, Claude-3) to analyze dialysis center data.
It provides comprehensive insights on center-wide performance, treatment quality, and patient outcomes based on the latest clinical guidelines.
""")

# Display center overview
st.header("Center Overview")

st.header("AI Analysis Results")
if st.button('Start Analysis'):
    prompt = generate_structured_prompt(lab_data, treatment_data)

    # Progress bar
    progress_bar = st.progress(0)
    total_analyses = 3  # GPT, Gemini, Claude
    progress_step = 1 / total_analyses

    tab1, tab2, tab3 = st.tabs(["GPT-4", "Gemini Pro", "Claude-3"])

    with tab1:
        st.subheader("GPT-4 Center Analysis")
        with st.spinner('GPT-4 analyzing, please wait...'):
            start_time_gpt = time.time()
            model_type, gpt_insights = generate_gpt_insights(prompt, "Dialysis Center Expert")
            end_time_gpt = time.time()
            st.info(f"GPT-4 analysis time: {end_time_gpt - start_time_gpt:.2f} seconds")
            st.markdown(gpt_insights)
            progress_bar.progress(1 * progress_step)

    with tab2:
        st.subheader("Gemini Pro Center Analysis")
        with st.spinner('Gemini Pro analyzing, please wait...'):    
            start_time_gemini = time.time()
            gemini_insights = generate_gemini_insights(prompt, "Dialysis Center Expert")
            end_time_gemini = time.time()
            st.info(f"Gemini Pro analysis time: {end_time_gemini - start_time_gemini:.2f} seconds")
            st.markdown(gemini_insights)
            progress_bar.progress(2 * progress_step)

    with tab3:
        st.subheader("Claude-3 Center Analysis")
        with st.spinner('Claude-3 analyzing, please wait...'):
            start_time_claude = time.time()
            claude_insights = generate_anthropic_insights(prompt, "Dialysis Center Expert")
            end_time_claude = time.time()
            st.info(f"Claude-3 analysis time: {end_time_claude - start_time_claude:.2f} seconds")
            st.markdown(claude_insights)
            progress_bar.progress(3 * progress_step)

else:
    st.info("Click 'Start Analysis' to begin.")

st.markdown("---")
st.markdown("© 2024 AI Dialysis Center Analysis Application. All rights reserved.")

