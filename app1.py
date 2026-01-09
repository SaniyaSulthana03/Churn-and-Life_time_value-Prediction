import streamlit as st
st.set_page_config(page_title="CLV & Churn Prediction", layout="centered")
st.title("💰 Customer Lifetime Value & Churn Prediction")
st.markdown("Predict CLV and churn risk using key behavioral metrics.")

st.header("Enter Customer Details")
avg_order_value = st.number_input("Average Order Value ($)", min_value=0.0, value=100.0)
total_purchases = st.number_input("Total Purchases", min_value=0, value=5)
membership_years = st.number_input("Membership Years", min_value=0.0, value=2.0)
days_since_last_purchase = st.number_input("Days Since Last Purchase", min_value=0, value=30)
cart_abandonment_rate = st.slider("Cart Abandonment Rate (%)", 0.0, 100.0, 20.0)
discount_usage_rate = st.slider("Discount Usage Rate (%)", 0.0, 100.0, 10.0)
returns_rate = st.slider("Returns Rate (%)", 0.0, 100.0, 5.0)
login_frequency = st.number_input("Login Frequency (per month)", min_value=0, value=15)
customer_service_calls = st.number_input("Customer Service Calls", min_value=0, value=2)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
country = st.selectbox("Country", ["USA","UK","France","India","Canada","Germany","Australia","Japan"])
signup_quarter = st.selectbox("Signup Quarter", ["Q1","Q2","Q3","Q4"])

input_df = pd.DataFrame([{
    'Average_Order_Value': avg_order_value,
    'Total_Purchases': total_purchases,
    'Membership_Years': membership_years,
    'Days_Since_Last_Purchase': days_since_last_purchase,
    'Cart_Abandonment_Rate': cart_abandonment_rate,
    'Discount_Usage_Rate': discount_usage_rate,
    'Returns_Rate': returns_rate,
    'Login_Frequency': login_frequency,
    'Customer_Service_Calls': customer_service_calls,
    'Gender': gender,
    'Country': country,
    'Signup_Quarter': signup_quarter
}])

if st.button("Predict CLV & Churn"):
    X_input = preprocessor.transform(input_df)
    predicted_clv = rf_clv.predict(X_input)[0]
    predicted_churn_prob = rf_churn.predict_proba(X_input)[:,1][0]
    
    if predicted_clv >= 2000 and predicted_churn_prob >= 0.5:
        segment = 'High-Value, High-Risk'
    elif predicted_clv >= 2000:
        segment = 'High-Value, Low-Risk'
    elif predicted_churn_prob >= 0.5:
        segment = 'Low-Value, High-Risk'
    else:
        segment = 'Low-Value, Low-Risk'
    
    st.subheader("Predictions 🎯")
    st.metric("Predicted CLV ($)", f"{predicted_clv:,.2f}")
    st.metric("Churn Probability (%)", f"{predicted_churn_prob*100:.1f}%")
    st.info(f"Customer Segment: **{segment}**")
    
    st.subheader("Top Factors Driving Prediction 🔑")
    st.write("- Average Order Value")
    st.write("- Total Purchases")
    st.write("- Engagement / Discount Sensitivity")
