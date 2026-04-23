import streamlit as st
import pandas as pd
import numpy as np
import pickle

@st.cache_resource
def load_artifacts():
    model        = pickle.load(open("financial_health_model.pkl", "rb"))
    columns      = pickle.load(open("columns.pkl", "rb"))
    percentiles  = pickle.load(open("percentiles.pkl", "rb"))
    return model, columns, percentiles

model, columns, percentiles = load_artifacts()

def to_percentile(value, lookup):
    """Convert a raw value to its percentile rank (0-1) using training data lookup."""
    return float(np.searchsorted(lookup, value)) / (len(lookup) - 1)

st.set_page_config(page_title="Financial Health Score", page_icon="💳")
st.title("💰 Financial Health Score Predictor")
st.write("Enter your details to see your financial health score:")

col1, col2 = st.columns(2)
with col1:
    age          = st.number_input("Age", 18, 100, value=35)
    income       = st.number_input("Monthly Income ($)", 0, 500_000, value=5000, step=500)
    debt_ratio   = st.number_input(
        "Debt Ratio (0 = no debt, 1 = all income to debt)",
        min_value=0.0, max_value=1.0, value=0.3, step=0.01
    )
with col2:
    credit_lines  = st.number_input("Open Credit Lines & Loans", 0, 58, value=5)
    late_payments = st.number_input("Times 90+ Days Late", 0, 98, value=0)
    dependents    = st.number_input("Number of Dependents", 0, 10, value=0)

if st.button("🔍 Predict My Score", use_container_width=True):

    # Convert user inputs to percentile scores using training data distribution
    income_pct = to_percentile(income,       percentiles["income"])
    debt_pct   = to_percentile(debt_ratio,   percentiles["debt"])
    credit_pct = to_percentile(credit_lines, percentiles["credit"])
    late_pct   = to_percentile(late_payments,percentiles["late"])

    # Build DataFrame with same feature names the model was trained on
    input_df = pd.DataFrame([{
        "age":                             age,
        "MonthlyIncome":                   income,
        "DebtRatio_clipped":               debt_ratio,
        "NumberOfOpenCreditLinesAndLoans": credit_lines,
        "NumberOfTimes90DaysLate":         late_payments,
        "NumberOfDependents":              dependents,
    }])[columns]

    prediction = float(model.predict(input_df)[0])
    prediction = max(0.0, min(100.0, prediction))

    st.markdown("---")
    st.subheader(f"Your Score: **{prediction:.1f} / 100**")
    st.progress(prediction / 100)

    if prediction > 75:
        st.success("🟢 Excellent Financial Health")
    elif prediction > 50:
        st.warning("🟡 Good Financial Health")
    elif prediction > 25:
        st.warning("🟠 Fair Financial Health")
    else:
        st.error("🔴 Poor Financial Health")

    # Show breakdown so user understands the score
    st.markdown("---")
    st.markdown("**Score Breakdown** (vs. others in dataset)")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Income",       f"{income_pct*100:.0f}th %ile")
    b2.metric("Low Debt",     f"{(1-debt_pct)*100:.0f}th %ile")
    b3.metric("Credit Lines", f"{credit_pct*100:.0f}th %ile")
    b4.metric("On-time Pay",  f"{(1-late_pct)*100:.0f}th %ile")
