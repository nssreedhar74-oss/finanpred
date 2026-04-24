import streamlit as st
import pandas as pd
import pickle

@st.cache_resource
def load_artifacts():
    model   = pickle.load(open("financial_health_model.pkl", "rb"))
    columns = pickle.load(open("columns.pkl", "rb"))
    return model, columns

model, columns = load_artifacts()

st.title("💰 Financial Health Score Predictor")
st.write("Enter your financial details to get your health score (0–100):")

col1, col2 = st.columns(2)
with col1:
    age          = st.number_input("Age", 18, 100, value=35)
    income       = st.number_input("Monthly Income ($)", 0, 200_000, value=5000, step=500)
    debt_ratio   = st.number_input(
        "Debt Ratio (0 = no debt · 1 = all income goes to debt)",
        min_value=0.0, max_value=1.0, value=0.3, step=0.01
    )
with col2:
    credit_lines  = st.number_input("Open Credit Lines & Loans", 0, 58, value=5)
    late_payments = st.number_input("Times 90+ Days Late", 0, 98, value=0)
    dependents    = st.number_input("Number of Dependents", 0, 10, value=0)

if st.button("🔍 Predict", use_container_width=True):
    input_data = pd.DataFrame([{
        "age":                             age,
        "MonthlyIncome":                   income,
        "DebtRatio_clipped":               debt_ratio,   # must match training column name
        "NumberOfOpenCreditLinesAndLoans": credit_lines,
        "NumberOfTimes90DaysLate":         late_payments,
        "NumberOfDependents":              dependents,
    }])[columns]   # enforce column order

    score = float(model.predict(input_data)[0])
    score = max(0.0, min(100.0, score))

    st.markdown("---")
    st.metric("💡 Financial Health Score", f"{score:.1f} / 100")
    st.progress(score / 100)

    if score > 75:
        st.success("🟢 Excellent Financial Health")
    elif score > 50:
        st.warning("🟡 Moderate Financial Health")
    else:
        st.error("🔴 Poor Financial Health")
