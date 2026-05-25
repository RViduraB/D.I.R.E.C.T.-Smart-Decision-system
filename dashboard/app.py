
import sys
import os
import streamlit as st
import pandas as pd

# Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from engine.main import run_project_flow 
from engine.predict import get_prediction # Integrating the predict.py module

st.set_page_config(page_title="D.I.R.E.C.T. System", layout="wide")

st.title("🌾 D.I.R.E.C.T. Smart Decision System")
st.markdown("### National Paddy & Rice Production Forecast")

# ---------------------------------------------------------
# SIDEBAR: FUTURE PREDICTION (2027)
# ---------------------------------------------------------
st.sidebar.header("Future Yield Predictor")
st.sidebar.info("Fill in the parameters to predict production.")

with st.sidebar.form("predict_form"):
    """
    Form block within the sidebar designed to collect user input features 
    required for generating future yield predictions.
    """
    predict_year = st.number_input("Year", min_value=2025, max_value=2030, value=2027)
    # Include districts and seed varieties that match your dataset configurations
    predict_dist = st.selectbox("District", ["Ampara", "Anuradhapura", "Polonnaruwa", "Kurunegala", "Hambantota"])
    predict_land = st.number_input("Land Area (Acres)", value=5.0)
    predict_seed = st.selectbox("Seed Variety", ["Bg352", "At362", "Bg300"])
    predict_rain = st.number_input("Expected Rainfall (mm)", value=1200.0)
    predict_ph = st.slider("Soil pH", 4.0, 9.0, 6.5)
    predict_pest = st.slider("Pest Damage (%)", 0.0, 100.0, 2.0)
    
    submit_btn = st.form_submit_button("Predict Yield Now")
    
    if submit_btn:
        """
        Triggers the underlying machine learning model prediction when the form is submitted.
        Calculates the expected paddy yield and converts it to its rice equivalent.
        """
        try:
            # Invoke the utility function from predict.py
            prediction = get_prediction(predict_year, predict_dist, predict_land, predict_seed, predict_rain, predict_ph, predict_pest)
            st.sidebar.success(f"**Predicted Yield:** {prediction:.2f} MT")
            st.sidebar.write(f"**Rice Equivalent:** {prediction * 0.68:.2f} MT")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

# ---------------------------------------------------------
# MAIN DASHBOARD: DATA FETCHING & VISUALIZATION
# ---------------------------------------------------------
with st.spinner('Activating D.I.R.E.C.T System...'):
    """
    Executes the analytical data processing pipeline (Bronze/Silver/Gold flow)
    and extracts structured results, overall totals, and demand projections.
    """
    data_list, total_prod, demand = run_project_flow()

if data_list:
    """
    Main dashboard visualization area executed upon successful data retrieval.
    Renders high-level KPIs, structured data tables, and distribution charts.
    """
    df = pd.DataFrame(data_list)

    # Metrics
    gap = total_prod - demand
    status = "Exportable Surplus" if gap > 0 else "Required Import"
    color = "#28a745" if gap > 0 else "#dc3545"

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rice Production", f"{total_prod:,.2f} MT")
    c2.metric("Annual Demand", f"{demand:,.2f} MT")
    c3.metric(status, f"{abs(gap):,.2f} MT")

    st.divider()

    # Detailed Table
    st.subheader("📋 District-wise Detailed Analysis")
    df_display = df.rename(columns={
        'Exp_Paddy': 'Expected Paddy (MT/Acre)',
        'Exp_Rice': 'Expected Rice (MT/Acre)',
        'Prod_Paddy': 'Total Paddy Production (MT)',
        'Prod_Rice': 'Total Rice Production (MT)'
    })
    st.dataframe(df_display, use_container_width=True)

    # Charts
    st.subheader("📊 Production Visualizer")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.write("**Total Paddy by District**")
        st.bar_chart(df.set_index('District')['Prod_Paddy'])

    with col_chart2:
        st.write("**Total Rice by District**")
        st.bar_chart(df.set_index('District')['Prod_Rice'])

    # Final Decision Box
    st.markdown(f"""
        <div style="background-color: {color}; padding: 25px; border-radius: 15px; text-align: center; color: white;">
            <h1>FINAL DECISION: {status.upper()}</h1>
            <h3>Quantity: {abs(gap):,.2f} Metric Tons</h3>
        </div>
    """, unsafe_allow_html=True)
else:
    st.error("Data could not be loaded. Please check your engine/main.py and CSV path.")