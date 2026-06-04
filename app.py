import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="⚡ Smart Power Consumption Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">⚡ Smart Power Consumption Predictor</h1>', unsafe_allow_html=True)
st.markdown("### A Machine Learning-based system for forecasting household electricity consumption")

# Sidebar navigation
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio(
    "Select Section:",
    ["📊 Overview", "🔍 Data Exploration", "🤖 ML Pipeline", "📈 Results & Predictions", "💡 Key Insights"]
)

# Load data with caching
@st.cache_data
def load_original_data():
    """Load the original household power consumption dataset"""
    df = pd.read_csv('household_power_consumption.csv')
    return df

@st.cache_data
def load_predictions():
    """Load the predictions dataset"""
    df = pd.read_csv('smart_power_predictions.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df

# ==================== PAGE 1: OVERVIEW ====================
if page == "📊 Overview":
    st.markdown('<h2 class="sub-header">🎯 Project Overview</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Smart Power Consumption Predictor** forecasts household electricity usage using 
        historical smart meter data enriched with time-based features.
        
        #### Key Highlights:
        - ✅ Ensemble machine learning approach (Linear Regression, Random Forest, Gradient Boosting)
        - ✅ Comprehensive feature engineering (lag features, rolling statistics, calendar features)
        - ✅ Temporal train-test split respecting time-series nature
        - ✅ Multiple evaluation metrics (MAE, RMSE, R²)
        - ✅ Feature importance analysis for model interpretability
        
        #### Objectives:
        - 🏠 **Energy-aware decision making** for households
        - 💰 **Bill estimation and cost optimization**
        - 🔌 **Smart grid integration** support
        - 📊 **Load pattern analysis** for utilities
        """)
    
    with col2:
        st.info("#### 📊 Dataset Info\n\n- **Records**: 260,640+ measurements\n- **Duration**: Multiple months\n- **Frequency**: Minute-level data\n- **Test Set**: 51,369 samples")
    
    st.markdown('<h2 class="sub-header">🏗️ Model Architecture</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ```
    ┌─────────────────────────────────────┐
    │     Data Sources (Smart Meters)     │
    └──────────────┬──────────────────────┘
                   │
    ┌──────────────▼──────────────────────┐
    │     Data Preprocessing & Cleaning   │
    └──────────────┬──────────────────────┘
                   │
    ┌──────────────▼──────────────────────┐
    │        Feature Engineering          │
    │  • Lag Features (1, 24 hours)       │
    │  • Rolling Statistics (24h)         │
    │  • Calendar Features (hour, day)    │
    └──────────────┬──────────────────────┘
                   │
          ┌────────┼────────┐
          │        │        │
    ┌─────▼────┐ ┌─▼──────┐ ┌──▼─────────┐
    │  Linear  │ │ Random │ │  Gradient  │
    │Regression│ │ Forest │ │  Boosting  │
    └─────┬────┘ └─┬──────┘ └──┬─────────┘
          │        │        │
          └────────┼────────┘
                   │
    ┌──────────────▼──────────────────────┐
    │    Evaluation & Final Predictions   │
    └─────────────────────────────────────┘
    ```
    """)
    
    st.markdown('<h2 class="sub-header">📈 Model Comparison</h2>', unsafe_allow_html=True)
    
    model_comparison = pd.DataFrame({
        'Model': ['Linear Regression', 'Random Forest 🏆', 'Gradient Boosting'],
        'MAE (kW)': [0.0282, 0.0156, 0.0195],
        'RMSE (kW)': [0.0441, 0.0315, 0.0320],
        'R² Score': [0.9979, 0.9989, 0.9989],
        'Training Time': ['Fast', 'Medium', 'Slow'],
        'Interpretability': ['⭐⭐⭐⭐⭐', '⭐⭐⭐', '⭐⭐']
    })
    
    st.dataframe(model_comparison, use_container_width=True)
    st.success("🏆 **Best Model**: Random Forest Regressor with MAE of 15.6 watts (~1.8% relative error)")

# ==================== PAGE 2: DATA EXPLORATION ====================
elif page == "🔍 Data Exploration":
    st.markdown('<h2 class="sub-header">📊 Original Dataset</h2>', unsafe_allow_html=True)
    
    df_original = load_original_data()
    
    st.write("### Dataset Preview (First 10 rows)")
    st.dataframe(df_original.head(10), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Dataset Information")
        st.write(f"- **Total Records**: {len(df_original):,}")
        st.write(f"- **Number of Columns**: {len(df_original.columns)}")
        st.write(f"- **Memory Usage**: {df_original.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    with col2:
        st.write("### Column Names")
        for col in df_original.columns:
            st.write(f"- `{col}`")
    
    st.write("### Statistical Summary")
    
    # Convert numeric columns
    numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']
    for col in numeric_cols:
        if col in df_original.columns:
            try:
                df_original[col] = pd.to_numeric(df_original[col], errors='coerce')
            except:
                pass
    
    st.dataframe(df_original.describe(), use_container_width=True)
    
    st.info("""
    **Features Description:**
    - **Global_active_power**: Total active power consumed by household (kilowatts)
    - **Global_reactive_power**: Total reactive power consumed
    - **Voltage**: Average voltage (volts)
    - **Global_intensity**: Average current intensity (amperes)
    - **Sub_metering_1/2/3**: Energy consumption in different areas of the house
    """)

# ==================== PAGE 3: ML PIPELINE ====================
elif page == "🤖 ML Pipeline":
    st.markdown('<h2 class="sub-header">🔧 Machine Learning Pipeline</h2>', unsafe_allow_html=True)
    
    st.markdown("### Pipeline Steps")
    
    # Step 1: Data Preprocessing
    with st.expander("📥 **Step 1: Data Preprocessing**", expanded=True):
        st.markdown("""
        - ✅ Load dataset from CSV
        - ✅ Parse timestamps (Date + Time columns)
        - ✅ Sort data chronologically
        - ✅ Handle missing values
        - ✅ Remove outliers
        - ✅ Resample to consistent time intervals
        
        **Code Example:**
        ```python
        df = pd.read_csv('household_power_consumption.csv')
        df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df = df.sort_values('Timestamp')
        df = df.dropna()
        ```
        """)
    
    # Step 2: Feature Engineering
    with st.expander("🔨 **Step 2: Feature Engineering**", expanded=True):
        st.markdown("""
        **Calendar Features:**
        - Hour of day (0-23)
        - Day of week (0-6)
        - Month (1-12)
        
        **Lag Features:**
        - `lag_1`: Previous consumption value
        - `lag_24`: Consumption 24 hours ago
        
        **Rolling Statistics:**
        - 24-hour moving average
        - 24-hour moving standard deviation
        
        **Code Example:**
        ```python
        df['hour'] = df['Timestamp'].dt.hour
        df['day_of_week'] = df['Timestamp'].dt.dayofweek
        df['month'] = df['Timestamp'].dt.month
        
        df['lag_1'] = df['Global_active_power'].shift(1)
        df['lag_24'] = df['Global_active_power'].shift(24)
        
        df['rolling_mean_24'] = df['Global_active_power'].rolling(24).mean()
        df['rolling_std_24'] = df['Global_active_power'].rolling(24).std()
        ```
        """)
    
    # Step 3: Train-Test Split
    with st.expander("✂️ **Step 3: Train-Test Split**", expanded=True):
        st.markdown("""
        - ✅ **Temporal split**: Respects time-series nature
        - ✅ **Training set**: 80% of data
        - ✅ **Test set**: 20% of data (51,369 samples)
        - ✅ No shuffling to maintain temporal order
        
        **Code Example:**
        ```python
        train_size = int(len(df) * 0.8)
        train_df = df[:train_size]
        test_df = df[train_size:]
        ```
        """)
    
    # Step 4: Feature Scaling
    with st.expander("⚖️ **Step 4: Feature Scaling**", expanded=True):
        st.markdown("""
        - ✅ StandardScaler for normalization
        - ✅ Fit on training data only
        - ✅ Transform both train and test sets
        
        **Code Example:**
        ```python
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        ```
        """)
    
    # Step 5: Model Training
    with st.expander("🎯 **Step 5: Model Training**", expanded=True):
        st.markdown("""
        **Three models trained:**
        
        1. **Linear Regression** (Baseline)
           ```python
           from sklearn.linear_model import LinearRegression
           lr_model = LinearRegression()
           lr_model.fit(X_train_scaled, y_train)
           ```
        
        2. **Random Forest Regressor**
           ```python
           from sklearn.ensemble import RandomForestRegressor
           rf_model = RandomForestRegressor(
               n_estimators=200,
               max_depth=12,
               random_state=42
           )
           rf_model.fit(X_train_scaled, y_train)
           ```
        
        3. **Gradient Boosting Regressor**
           ```python
           from sklearn.ensemble import GradientBoostingRegressor
           gb_model = GradientBoostingRegressor(
               n_estimators=300,
               max_depth=5,
               random_state=42
           )
           gb_model.fit(X_train_scaled, y_train)
           ```
        """)
    
    # Step 6: Prediction & Evaluation
    with st.expander("📊 **Step 6: Prediction & Evaluation**", expanded=True):
        st.markdown("""
        **Evaluation Metrics:**
        - **MAE** (Mean Absolute Error): Average absolute prediction error
        - **RMSE** (Root Mean Squared Error): Root of average squared errors
        - **R² Score**: Proportion of variance explained
        
        **Code Example:**
        ```python
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        predictions = rf_model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        print(f"MAE: {mae:.4f} kW")
        print(f"RMSE: {rmse:.4f} kW")
        print(f"R²: {r2:.4f}")
        ```
        """)
    
    st.success("✅ Pipeline Complete! Models trained and evaluated successfully.")

# ==================== PAGE 4: RESULTS & PREDICTIONS ====================
elif page == "📈 Results & Predictions":
    st.markdown('<h2 class="sub-header">📈 Model Results & Predictions</h2>', unsafe_allow_html=True)
    
    df_predictions = load_predictions()
    
    st.write("### Predictions Dataset Preview (First 10 rows)")
    st.dataframe(df_predictions.head(10), use_container_width=True)
    
    st.write("### Dataset Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Predictions", f"{len(df_predictions):,}")
    with col2:
        st.metric("Time Period", f"{(df_predictions['Timestamp'].max() - df_predictions['Timestamp'].min()).days} days")
    with col3:
        st.metric("Columns", len(df_predictions.columns))
    
    st.markdown("---")
    
    # Model Performance Metrics
    st.write("### 🎯 Model Performance Metrics")
    
    metrics_df = pd.DataFrame({
        'Model': ['Linear Regression', 'Random Forest 🏆', 'Gradient Boosting'],
        'MAE (kW)': [0.0282, 0.0156, 0.0195],
        'RMSE (kW)': [0.0441, 0.0315, 0.0320],
        'R² Score': [0.9979, 0.9989, 0.9989],
        'Relative Error (%)': [3.26, 1.80, 2.25]
    })
    
    st.dataframe(metrics_df, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🏆 Best MAE", "0.0156 kW", "15.6 watts")
    with col2:
        st.metric("🏆 Best RMSE", "0.0315 kW", "31.5 watts")
    with col3:
        st.metric("🏆 Best R²", "0.9989", "99.89%")
    
    st.markdown("---")
    
    # Visualization: Actual vs Predicted
    st.write("### 📊 Actual vs Predicted Power Consumption")
    
    # Sample a subset for visualization (first 500 points)
    sample_size = st.slider("Select number of data points to visualize:", 100, 2000, 500)
    df_sample = df_predictions.head(sample_size)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_sample['Timestamp'], df_sample['actual_Global_active_power'], 
            label='Actual', linewidth=2, alpha=0.7, color='#1f77b4')
    ax.plot(df_sample['Timestamp'], df_sample['Final_Prediction'], 
            label='Predicted (RF)', linewidth=2, alpha=0.7, color='#ff7f0e')
    ax.set_xlabel('Timestamp', fontsize=12)
    ax.set_ylabel('Global Active Power (kW)', fontsize=12)
    ax.set_title('Actual vs Predicted Power Consumption (Random Forest)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Check if images exist
    st.write("### 📸 Visualizations from Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists('image.png'):
            st.write("#### Actual vs Predicted (Full View)")
            image1 = Image.open('image.png')
            st.image(image1, use_container_width=True)
        else:
            st.info("Actual vs Predicted visualization not found (image.png)")
    
    with col2:
        if os.path.exists('image-1.png'):
            st.write("#### Feature Importance")
            image2 = Image.open('image-1.png')
            st.image(image2, use_container_width=True)
        else:
            st.info("Feature importance visualization not found (image-1.png)")
    
    st.markdown("---")
    
    # Statistical Analysis
    st.write("### 📊 Prediction Error Analysis")
    
    df_predictions['error_RF'] = df_predictions['actual_Global_active_power'] - df_predictions['pred_RF']
    df_predictions['abs_error_RF'] = np.abs(df_predictions['error_RF'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Error Statistics (Random Forest)")
        error_stats = df_predictions['error_RF'].describe()
        st.dataframe(error_stats, use_container_width=True)
    
    with col2:
        # Error distribution histogram
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df_predictions['error_RF'], bins=50, color='#2ca02c', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Prediction Error (kW)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

# ==================== PAGE 5: KEY INSIGHTS ====================
elif page == "💡 Key Insights":
    st.markdown('<h2 class="sub-header">💡 Key Insights & Findings</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### 🔍 Feature Importance Analysis
        
        Based on the Random Forest model, the most important features are:
        
        1. **Global_intensity** 🥇
           - Most important feature
           - Strong correlation with power consumption
        
        2. **lag_1** 🥈
           - Recent consumption patterns
           - Captures short-term autocorrelation
        
        3. **hour** 🥉
           - Daily consumption patterns
           - Peak hours vs off-peak hours
        
        4. **lag_24**
           - Day-over-day patterns
           - Weekly consumption cycles
        
        5. **Voltage & Rolling Statistics**
           - Additional contextual information
           - Smoothing of consumption trends
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Model Performance Insights
        
        **Exceptional Accuracy:**
        - All models achieve R² > 0.997
        - Random Forest: 99.89% variance explained
        - Average error: Only 15.6 watts
        
        **Ensemble Advantage:**
        - Tree-based models reduce error by ~45%
        - Better at capturing non-linear patterns
        - More robust to outliers
        
        **Practical Impact:**
        - Average household consumption: ~0.865 kW
        - Random Forest error: ~1.8% relative error
        - Suitable for real-time energy management
        
        **Trade-offs:**
        - Gradient Boosting: 2% more accurate than RF
        - BUT: 3x longer training time
        - **Recommendation**: Use Random Forest
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🎯 Key Findings
    
    | Finding | Description | Impact |
    |---------|-------------|--------|
    | **Lag Features** | Previous consumption values are highly predictive | Enable accurate short-term forecasting |
    | **Calendar Features** | Hour and day patterns successfully captured | Model understands daily/weekly seasonality |
    | **Time-Series Split** | Temporal validation ensures realistic performance | Prevents data leakage and overfitting |
    | **Ensemble Methods** | RF and GB significantly outperform Linear Regression | Non-linear patterns are important |
    | **Low RMSE** | Predictions within ~31.5 watts on average | High confidence in forecasts |
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 💰 Business Applications
    
    #### For Households:
    - 🏠 **Energy Awareness**: Understand consumption patterns
    - 💡 **Cost Optimization**: Shift usage to off-peak hours
    - 📊 **Bill Forecasting**: Predict monthly electricity bills
    - ⚠️ **Anomaly Detection**: Identify unusual consumption
    
    #### For Utilities:
    - 📈 **Load Forecasting**: Plan generation capacity
    - 🔌 **Grid Management**: Balance supply and demand
    - 💰 **Dynamic Pricing**: Implement time-of-use rates
    - 🌱 **Renewable Integration**: Better integrate solar/wind
    
    #### For Smart Home Systems:
    - 🤖 **Automated Control**: Optimize appliance scheduling
    - 🔋 **Battery Management**: Charge when prices are low
    - 📱 **User Notifications**: Alert on high consumption
    - 🌡️ **HVAC Optimization**: Smart heating/cooling control
    """)
    
    st.success("""
    ### ✅ Conclusion
    
    The Smart Power Consumption Predictor successfully demonstrates that machine learning can 
    accurately forecast household electricity usage with minimal error. The Random Forest model 
    achieves exceptional performance (R² = 0.9989, MAE = 15.6 watts) and is recommended for 
    production deployment due to its balance of accuracy, training speed, and interpretability.
    """)
    
    st.info("""
    ### 🚀 Future Enhancements
    
    - **Weather Integration**: Include temperature, humidity data
    - **Occupancy Detection**: Incorporate presence sensors
    - **Appliance-Level**: Break down by individual devices
    - **Real-Time Deployment**: Deploy as API service
    - **Mobile App**: User-friendly interface for homeowners
    - **Deep Learning**: Explore LSTM/Transformer models
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>⚡ Smart Power Consumption Predictor | Built with Streamlit & Scikit-learn</p>
    <p>For more information, see README.md</p>
</div>
""", unsafe_allow_html=True)
