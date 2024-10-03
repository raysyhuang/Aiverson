import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from plotly.subplots import make_subplots

# Define feature types globally
numeric_features = ['PreWeight', 'WeightChange', 'UltrafiltrationVolume', 'PreSystolic', 'PreDiastolic', 
                    'PrePulse'] + [f'{col}_lag' for col in ['UricAcid', 'Creatinine', 'PreDialysisUreaNitrogen', 
                                                            'Sodium', 'Potassium', 'Albumin', 'Hemoglobin']]
categorical_features = ['DialysisType']

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Split BP into systolic and diastolic
    df[['PreSystolic', 'PreDiastolic']] = df['PreBP'].str.split('/', expand=True).astype(int)
    df[['PostSystolic', 'PostDiastolic']] = df['PostBP'].str.split('/', expand=True).astype(int)
    
    # Create lag features for lab values
    lab_columns = ['UricAcid', 'Creatinine', 'PreDialysisUreaNitrogen', 'Sodium', 'Potassium', 'Albumin', 'Hemoglobin']
    for col in lab_columns:
        df[f'{col}_lag'] = df.groupby('TreatmentNumber')[col].shift(1)
    
    # Select features for the model
    features = ['PreWeight', 'WeightChange', 'UltrafiltrationVolume', 'PreSystolic', 'PreDiastolic', 
                'PrePulse', 'DialysisType'] + [f'{col}_lag' for col in lab_columns]
    
    X = df[features]
    y = df['PostSystolic']
    
    return df, X, y

# Train the model
@st.cache_resource
def train_outcome_predictor(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define preprocessing steps for numeric and categorical columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', sparse=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create pipelines for different models
    rf_model = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
    
    lr_model = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', LinearRegression())])
    
    gb_model = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', GradientBoostingRegressor(random_state=42))])

    models = [rf_model, lr_model, gb_model]
    model_names = ['Random Forest', 'Linear Regression', 'Gradient Boosting']

    # Perform cross-validation for each model
    scores = []
    for model in models:
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        scores.append(cv_scores.mean())

    # Select the best model
    best_model_index = np.argmax(scores)
    best_model = models[best_model_index]
    best_model_name = model_names[best_model_index]
    best_score = scores[best_model_index]

    # Fit the best model on the entire dataset
    best_model.fit(X, y)

    return best_model, best_score, best_model_name, numeric_features, categorical_features

# Streamlit app
st.title('Dialysis Outcome Predictor')

# Load and preprocess data
df, X, y = load_and_preprocess_data("files/xiaolong-dialysis-dataset.csv")

# Train model
model, score, model_name, num_features, cat_features = train_outcome_predictor(X, y)

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Dataset Overview", "Data Visualization", "Prediction Model"])

if page == "Dataset Overview":
    st.header("Dataset Overview")
    st.write(df)
    st.write(f"Dataset shape: {df.shape}")
    
    st.subheader("Summary Statistics")
    st.write(df.describe())
    
    st.subheader("Missing Values")
    missing_values = df.isnull().sum()
    st.write(missing_values[missing_values > 0])

elif page == "Data Visualization":
    st.header("Data Visualization")
    
    st.subheader("Weight Change Distribution")
    fig = px.histogram(df, x='WeightChange', nbins=20, marginal='box')
    st.plotly_chart(fig)
    
    st.subheader("Pre vs Post Systolic Blood Pressure")
    fig = px.scatter(df, x='PreSystolic', y='PostSystolic', trendline='ols')
    fig.add_trace(go.Scatter(x=[df['PreSystolic'].min(), df['PreSystolic'].max()], 
                             y=[df['PreSystolic'].min(), df['PreSystolic'].max()],
                             mode='lines', name='y=x', line=dict(color='red', dash='dash')))
    st.plotly_chart(fig)
    
    st.subheader("Correlation Heatmap")
    corr_matrix = df[['PreWeight', 'PostWeight', 'WeightChange', 'UltrafiltrationVolume', 
                      'PreSystolic', 'PostSystolic', 'PreDiastolic', 'PostDiastolic']].corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto")
    st.plotly_chart(fig)

    # New visualization: Pre and Post BP and Pulse over time
    st.subheader("Pre and Post BP and Pulse Over Time")
    
    # Create a new dataframe with the required columns
    plot_df = df[['Date', 'PreSystolic', 'PostSystolic', 'PreDiastolic', 'PostDiastolic', 'PrePulse', 'PostPulse']]
    
    # Ensure the Date column is in datetime format
    plot_df['Date'] = pd.to_datetime(plot_df['Date'])
    
    # Sort the dataframe by date
    plot_df = plot_df.sort_values('Date')
    
    # Create subplots: one for BP, one for Pulse
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Blood Pressure Over Time", "Pulse Over Time"))
    
    # Add traces for BP
    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['PreSystolic'], mode='lines', name='Pre Systolic'), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['PostSystolic'], mode='lines', name='Post Systolic'), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['PreDiastolic'], mode='lines', name='Pre Diastolic'), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['PostDiastolic'], mode='lines', name='Post Diastolic'), row=1, col=1)
    
    # Add traces for Pulse
    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['PrePulse'], mode='lines', name='Pre Pulse'), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['PostPulse'], mode='lines', name='Post Pulse'), row=2, col=1)
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text='Pre and Post BP and Pulse Over Time',
        hovermode='x unified'
    )
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Blood Pressure (mmHg)", row=1, col=1)
    fig.update_yaxes(title_text="Pulse (bpm)", row=2, col=1)
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

elif page == "Prediction Model":
    st.header("Prediction Model")
    st.write(f"Best Model: {model_name}")
    st.write(f"Model R-squared score: {score:.4f}")
    
    st.subheader("Feature Importance")
    if hasattr(model['regressor'], 'feature_importances_'):
        # Get feature names after preprocessing
        feature_names = (model['preprocessor']
                         .named_transformers_['num']
                         .get_feature_names_out(num_features).tolist() +
                         model['preprocessor']
                         .named_transformers_['cat']
                         .get_feature_names_out(cat_features).tolist())
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model['regressor'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig = px.bar(feature_importance.head(10), x='importance', y='feature', orientation='h',
                     title='Top 10 Most Important Features')
        st.plotly_chart(fig)
    else:
        st.write("Feature importance is not available for this model.")
    
    st.subheader("Make a Prediction")
    col1, col2 = st.columns(2)
    with col1:
        pre_weight = st.number_input('Pre-dialysis Weight (kg)', value=58.0)
        weight_change = st.number_input('Weight Change (kg)', value=-3.5)
        uf_volume = st.number_input('Ultrafiltration Volume (L)', value=3.8)
        pre_systolic = st.number_input('Pre-dialysis Systolic BP', value=130)
    with col2:
        pre_diastolic = st.number_input('Pre-dialysis Diastolic BP', value=80)
        pre_pulse = st.number_input('Pre-dialysis Pulse', value=75)
        dialysis_type = st.selectbox('Dialysis Type', ['HD', 'HDF'])
    
    if st.button('Predict'):
        new_treatment = pd.DataFrame({
            'PreWeight': [pre_weight], 'WeightChange': [weight_change], 
            'UltrafiltrationVolume': [uf_volume],
            'PreSystolic': [pre_systolic], 'PreDiastolic': [pre_diastolic], 
            'PrePulse': [pre_pulse], 'DialysisType': [dialysis_type],
            'UricAcid_lag': [390], 'Creatinine_lag': [1300], 
            'PreDialysisUreaNitrogen_lag': [35],
            'Sodium_lag': [140], 'Potassium_lag': [4.1], 
            'Albumin_lag': [38], 'Hemoglobin_lag': [105]
        })
        
        prediction = model.predict(new_treatment)
        
        st.write(f"Predicted post-dialysis systolic BP: {prediction[0]:.2f}")

st.sidebar.info("This app demonstrates a dialysis outcome predictor based on treatment data and lab results. Use the navigation menu to explore the dataset, visualizations, and make predictions.")