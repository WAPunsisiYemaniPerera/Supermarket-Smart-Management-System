import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from datetime import datetime

# --- App Configuration ---
st.set_page_config(
    page_title="Cargills Sales & Demand Predictor",
    page_icon="🛒",
    layout="wide"
)

# --- Load Models and Data ---
@st.cache_data
def load_data():
    """Loads and prepares the necessary data files."""
    try:
        # CORRECTED: Use relative paths for portability
        df_original = pd.read_csv('C:/Users/3C Tech/Desktop/supermarket smart managment sytem/dataset/cargills.csv')
        df_original['Date'] = pd.to_datetime(df_original['Date'])
        return df_original
    except FileNotFoundError:
        st.error("Error: 'cargills.csv' not found. Please place it in the same directory as the app.")
        return None

@st.cache_resource
def load_models():
    """Loads all pre-trained models and scalers."""
    try:
        # Use relative paths
        with open('C:/Users/3C Tech/Desktop/supermarket smart managment sytem/deployment/best_profit_model.pkl', 'rb') as file:
            profit_model = pickle.load(file)
        # Load the new, advanced loyalty model and scaler
        with open('C:/Users/3C Tech/Desktop/supermarket smart managment sytem/deployment/best_loyalty_model_advanced.pkl', 'rb') as file:
            loyalty_model = pickle.load(file)
        with open('C:/Users/3C Tech/Desktop/supermarket smart managment sytem/deployment/loyalty_model_scaler_advanced.pkl', 'rb') as file:
            loyalty_scaler = pickle.load(file)
        return profit_model, loyalty_model, loyalty_scaler
    except FileNotFoundError as e:
        st.error(f"Error loading model file: {e}. Please ensure all .pkl files are in the same directory as the app.")
        return None, None, None

# Load the resources
df_original = load_data()
profit_model, loyalty_model, loyalty_scaler = load_models()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dashboard", "Profit Predictor", "Demand Forecaster", "Loyalty Insights"])
st.sidebar.markdown("---")
st.sidebar.info(
    "This application provides predictive insights into supermarket sales. "
    "Use the tools to forecast profit, analyze item demand, and identify potential loyalty members."
)

# =====================================================================================
# HOME PAGE
# =====================================================================================
def home_page():
    st.title("Welcome to the Supermarket Smart Management System 🛒")
    st.markdown("#### An Interactive Tool for Data-Driven Retail Insights")
    st.markdown("This application leverages machine learning to analyze and predict supermarket sales data, transforming raw numbers into actionable business intelligence.")
    st.markdown("---")

    st.subheader("Application Modules Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Dashboard & Data Exploration**")
        st.markdown("""
        - **Interactive Dashboard:** Get a high-level overview of key metrics.
        - **Dynamic Filtering:** Filter the entire dataset by outlet and department.
        - **Multiple Visualizations:** Explore data through various chart types.
        """)
    with col2:
        st.success("**Predictive Modeling**")
        st.markdown("""
        - **Profit Predictor:** Forecast the profit of a single transaction.
        - **Loyalty Insights:** Predict a customer's likelihood of joining the loyalty program.
        - **Demand Forecaster:** Identify best-selling and trending products.
        """)

    st.markdown("---")
    st.subheader("Model Performance Metrics")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Profit Prediction Model (Regression)**")
        st.metric(label="R-squared (R²)", value="0.887")
        st.metric(label="Root Mean Squared Error (RMSE)", value="Rs. 83.08")
        # --- REPLACED st.help with st.markdown ---
        st.markdown("> *This model explains ~89% of the variance in profit and is typically accurate within Rs. 83.*")
    with c2:
        st.markdown("**Loyalty Prediction Model (Classification)**")
        st.metric(label="Accuracy", value="55.2%")
        st.metric(label="ROC AUC", value="0.55")
        # --- REPLACED st.help with st.markdown ---
        st.markdown("> *This model identifies potential loyalty members. An ROC AUC of 0.55 indicates it has slight predictive power, slightly better than a random guess.*")

# =====================================================================================
# DASHBOARD PAGE
# =====================================================================================
def dashboard_page():
    st.title("📊 Sales Dashboard")
    st.markdown("An interactive overview of key performance metrics. Use the filters below to drill down into the data.")

    if df_original is None: return

    # --- Interactive Filters ---
    col1, col2 = st.columns(2)
    with col1:
        outlet_filter = st.multiselect(
            "Select Outlet(s):",
            options=df_original['Outlet_Name'].unique(),
            default=df_original['Outlet_Name'].unique()
        )
    with col2:
        department_filter = st.multiselect(
            "Select Department(s):",
            options=df_original['Department'].unique(),
            default=df_original['Department'].unique()
        )

    filtered_df = df_original[
        (df_original['Outlet_Name'].isin(outlet_filter)) & 
        (df_original['Department'].isin(department_filter))
    ]
    st.markdown("---")

    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        return

    # --- Key Metrics ---
    total_profit = filtered_df['Profit'].sum()
    total_sales_value = filtered_df['TotalValue'].sum()
    avg_transaction_value = filtered_df['TotalValue'].mean()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Profit", f"Rs. {total_profit:,.0f}")
    col2.metric("Total Sales Value", f"Rs. {total_sales_value:,.0f}")
    col3.metric("Avg. Transaction Value", f"Rs. {avg_transaction_value:,.2f}")
    st.markdown("---")
    
    # --- Visualizations ---
    st.subheader("Performance Analysis")
    daily_profit = filtered_df.groupby('Date')['Profit'].sum()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily_profit.index, daily_profit.values, color='royalblue', marker='o', linestyle='-', markersize=4)
    ax.set_title("Daily Profit Over Time (Filtered)")
    ax.set_ylabel("Total Profit (Rs.)")
    st.pyplot(fig)

# =====================================================================================
# PROFIT PREDICTOR PAGE
# =====================================================================================
def profit_predictor_page():
    st.title("🛒 Profit Predictor")
    st.markdown("Predict the profit for a single transaction using a pre-trained **Gradient Boosting** model.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        prediction_date = st.date_input("Date of Transaction", datetime.now())
        unit_price = st.number_input("Unit Price (Rs.)", min_value=0.0, value=500.0, step=10.0)
        quantity = st.number_input("Quantity Sold", min_value=1, value=2)
    with col2:
        discount = st.slider("Discount (%)", min_value=0, max_value=100, value=10, step=5)
        age = st.slider("Customer Age", min_value=18, max_value=80, value=35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        nexus_member = st.selectbox("Nexus Member?", ["Yes", "No"])
    with col3:
        outlet_name = st.selectbox("Outlet Name", df_original['Outlet_Name'].unique())
        department = st.selectbox("Department", df_original['Department'].unique())
        payment_method = st.selectbox("Payment Method", df_original['Payment_Method'].unique())

    if st.button("Predict Profit", type="primary"):
        if profit_model is None: return
        
        total_value = (unit_price * quantity) * (1 - discount / 100)
        date = pd.to_datetime(prediction_date)
        input_data = {
            'Age': age, 'NexusMember': 1 if nexus_member == "Yes" else 0,
            'Quantity': quantity, 'Unit_Price': unit_price, 'Discount': discount,
            'TotalValue': total_value, 'Loyalty_Points_Earned': 0,
            'Year': date.year, 'Month': date.month, 'Day': date.day,
            'DayOfWeek': date.dayofweek, 'WeekOfYear': date.isocalendar().week
        }
        input_data['Gender_Male'] = 1 if gender == "Male" else 0
        input_data['Gender_Female'] = 1 if gender == "Female" else 0
        
        model_features = profit_model.feature_names_in_
        # Create a DataFrame from the dictionary and ensure column order matches the model
        input_df = pd.DataFrame([input_data])
        for feature in model_features:
            if 'Outlet_Name_' in feature and feature.endswith(outlet_name): input_df[feature] = 1
            elif 'Department_' in feature and feature.endswith(department): input_df[feature] = 1
            elif 'Payment_Method_' in feature and feature.endswith(payment_method): input_df[feature] = 1
            elif feature not in input_df.columns: input_df[feature] = 0
        
        input_df = input_df[model_features] # Enforce column order
        prediction = profit_model.predict(input_df)
        
        st.markdown("---")
        st.success(f"**Predicted Profit:** Rs. {prediction[0]:,.2f}")

# =====================================================================================
# DEMAND FORECASTER PAGE
# =====================================================================================
@st.cache_data
def run_demand_analysis(df, outlet, department):
    if outlet != "All": df = df[df['Outlet_Name'] == outlet]
    if department != "All": df = df[df['Department'] == department]
    if df.empty: return pd.DataFrame(), pd.Series(dtype='float64'), pd.DataFrame()
    df_ts = df.set_index('Date')
    weekly_sales = df_ts.groupby('Item_Description').resample('W')['Quantity'].sum().reset_index()
    weekly_sales = weekly_sales[weekly_sales['Quantity'] > 0]
    total_sales = df.groupby('Item_Description')['Quantity'].sum().sort_values(ascending=False)
    trend_data = []
    for item in weekly_sales['Item_Description'].unique():
        item_sales = weekly_sales[weekly_sales['Item_Description'] == item].copy()
        if len(item_sales) >= 3:
            item_sales['time_index'] = np.arange(len(item_sales))
            slope, _, _, _, _ = linregress(item_sales['time_index'], item_sales['Quantity'])
            trend_data.append({'Item_Description': item, 'Total_Quantity_Sold': total_sales.loc[item], 'Trend_Score': round(slope, 4)})
    return pd.DataFrame(trend_data), total_sales, weekly_sales

def demand_forecaster_page():
    st.title("📈 Demand Forecaster")
    st.markdown("Analyze historical sales to identify which items will likely be in high demand.")
    if df_original is None: return
    
    col1, col2 = st.columns(2)
    with col1: outlet_filter = st.selectbox("Filter by Outlet", ["All"] + list(df_original['Outlet_Name'].unique()))
    with col2: department_filter = st.selectbox("Filter by Department", ["All"] + list(df_original['Department'].unique()))
    trend_df, total_sales, weekly_sales = run_demand_analysis(df_original, outlet_filter, department_filter)
    st.markdown("---")
    if trend_df.empty:
        st.warning("No sales data available for the selected filters.")
        return
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("✅ The Sure Bets")
        sure_bets = trend_df[trend_df['Trend_Score'].abs() < 0.05].sort_values(by='Total_Quantity_Sold', ascending=False)
        st.dataframe(sure_bets.head(5), use_container_width=True)
    with col2:
        st.subheader("⭐ The Rising Stars")
        rising_stars = trend_df[trend_df['Trend_Score'] > 0].sort_values(by='Trend_Score', ascending=False)
        st.dataframe(rising_stars.head(5), use_container_width=True)
    st.markdown("---")
    st.subheader("🔍 Specific Item Analysis")
    if not total_sales.empty:
        selected_item = st.selectbox("Select an item to analyze its trend", sorted(total_sales.index))
        item_weekly_sales = weekly_sales[weekly_sales['Item_Description'] == selected_item]
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=item_weekly_sales, x='Date', y='Quantity', ax=ax, marker='o')
        st.pyplot(fig)

# =====================================================================================
# LOYALTY INSIGHTS PAGE (CORRECTED)
# =====================================================================================
def loyalty_insights_page():
    st.title("🎯 Loyalty Insights")
    st.markdown("Identify customers who are strong candidates for the **Nexus Loyalty Program** using our tuned **Logistic Regression** model.")
    st.markdown("---")

    if loyalty_model is None or loyalty_scaler is None:
        st.error("Loyalty model or scaler could not be loaded. Please check the files.")
        return

    st.subheader("Enter Transaction Details to Predict Loyalty Potential")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        transaction_date = st.date_input("Date of Transaction", datetime.now())
        age = st.slider("Customer Age", 18, 80, 35)
        gender = st.selectbox("Gender", df_original['Gender'].unique())

    with col2:
        department = st.selectbox("Department", df_original['Department'].unique())
        payment_method = st.selectbox("Payment Method", df_original['Payment_Method'].unique())
        quantity = st.number_input("Number of Items", 1, 100, 5)

    with col3:
        unit_price = st.number_input("Average Price per Item (Rs.)", 10.0, 10000.0, 500.0, step=50.0)
        discount = st.slider("Total Discount Given (Rs.)", 0, 5000, 250, step=50)
        total_value = st.number_input("Final Total Value (Rs.)", 10.0, 50000.0, 2250.0, step=100.0)


    if st.button("Predict Loyalty Potential", type="primary"):
        input_data = {}

        date = pd.to_datetime(transaction_date)
        input_data['Age'] = age
        input_data['Quantity'] = quantity
        input_data['Discount'] = discount
        input_data['TotalValue'] = total_value
        input_data['PricePerUnit'] = unit_price
        input_data['TimeOfDay_Hour'] = 14
        input_data['IsWeekend'] = 1 if date.dayofweek >= 5 else 0
        
        original_total = (quantity * unit_price)
        input_data['DiscountRate'] = (discount / original_total) * 100 if original_total > 0 else 0

        input_df = pd.DataFrame([input_data])

        # Manually create the one-hot encoded columns from user selections
        input_df[f'Gender_{gender}'] = 1
        input_df[f'Department_{department}'] = 1
        input_df[f'Payment_Method_{payment_method}'] = 1
        input_df[f'DayOfWeek_{date.day_name()}'] = 1
        input_df[f'Month_{date.strftime("%B")}'] = 1
        
        # Get the exact feature list the model was trained on from the SCALER
        training_cols = loyalty_scaler.get_feature_names_out()
        
        # Reindex the input DataFrame to match the training data columns exactly
        input_df_reindexed = input_df.reindex(columns=training_cols, fill_value=0)
        
        # Scale the entire, correctly ordered DataFrame. The scaler knows what to do.
        input_scaled = loyalty_scaler.transform(input_df_reindexed)

        # Make the prediction
        prediction = loyalty_model.predict(input_scaled)
        probability = loyalty_model.predict_proba(input_scaled)

        st.markdown("---")
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.success(f"**Conclusion: This customer is a STRONG candidate for the loyalty program.**")
            st.progress(float(probability[0][1]))
            st.metric(label="Probability of Becoming a Member", value=f"{probability[0][1]:.1%}")
        else:
            st.warning(f"**Conclusion: This customer is NOT a likely candidate for the loyalty program.**")
            st.progress(float(probability[0][1]))
            st.metric(label="Probability of Becoming a Member", value=f"{probability[0][1]:.1%}")

# --- Main App Logic ---
if df_original is not None:
    if page == "Home":
        home_page()
    elif page == "Dashboard":
        dashboard_page()
    elif page == "Profit Predictor":
        profit_predictor_page()
    elif page == "Demand Forecaster":
        demand_forecaster_page()
    elif page == "Loyalty Insights":
        loyalty_insights_page()
