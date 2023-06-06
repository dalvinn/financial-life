import streamlit as st

import sys

sys.path.append("src")
import utilities as utils
import parameters as params
from squigglepy import K, M

# Configurations
st.set_page_config(
    page_title="Financial Life App", 
    #layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
    )

#st.markdown(
#    """
#<style>
#body {
#    color: #fff;
#    background-color: #4F8BF9;
#}
#</style>
#    """, 
#    unsafe_allow_html=True
#    )
#
#st.markdown("""
#<style>
#    h1 {
#        color: red;
#    }
#</style>
#""", unsafe_allow_html=True)

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

st.title("Financial Life Model")

# Create a first row of sliders
row1_space1, row1_space2 = st.columns(2)

with row1_space1:
    m = st.slider("Number of paths", 100, 500, 300)
    years = st.slider("Number of years", 10, 80, 60)
    income_fraction_consumed = st.slider("income_fraction_consumed", 0.0, 2.0, 0.1)
    wealth_fraction_consumed_before_retirement = st.slider("wealth_fraction_consumed_before_retirement", 0.0, 2.0, 1.0)
    wealth_fraction_consumed_after_retirement = st.slider("wealth_fraction_consumed_after_retirement", 0.0, 2.0, 1.0)
    max_cash_threshold = st.slider("max_cash_threshold", 3_000, 30_000, 5_000)
    min_cash_threshold = st.slider("min_cash_threshold", 0, 10_000, 5_000)

with row1_space2:
    cash_start = st.slider("Initial cash", 0, 100_000, 10_000)
    market_start = st.slider("Initial market wealth", 0, 500_000, 30_000)
    income_start = st.slider("Initial income", 20_000, 200_000, 40_000)

# Create a collapsible section for the second row
with st.expander("Advanced Settings"):
    row2_space1, row2_space2 = st.columns(2)

    with row2_space1:
        min_income = st.slider(
            "Minimum income",
            0,
            50_000,
            10_000,
            help="The reservation income, a lower bound that you don't expect to dip below.",
        )
        inflation_rate = st.slider("Inflation rate", 0.0, 0.1, 0.02)

    with row2_space2:
        r = st.slider(
            "Interest rate", 0.0, 0.1, 0.02, help="The rate of return on savings."
        )

variables_to_plot = st.multiselect(
    "Select variables to plot",
    options=[
        "income",
        "cash",
        "market",
        "consumption",
        "inflation",
        "savings",
        "financial_wealth",
    ],
    default=["income", "cash", "market", "consumption"],
)

# Construct the parameters dictionary
input_params = params.input_params
input_params["m"] = m
input_params["years"] = years + 1
input_params["income_fraction_consumed"] = income_fraction_consumed
input_params["wealth_fraction_consumed_before_retirement"] = wealth_fraction_consumed_before_retirement
input_params["wealth_fraction_consumed_after_retirement"] = wealth_fraction_consumed_after_retirement
input_params["max_cash_threshold"] = max_cash_threshold
input_params["min_cash_threshold"] = min_cash_threshold
input_params["cash_start"] = cash_start
input_params["market_start"] = market_start
input_params["life_cycle_income"] = [
    income_start + (year * 2.5 * 1000) for year in range(input_params["years"])
]
input_params["min_income"] = min_income
input_params["inflation_rate"] = inflation_rate
input_params["r"] = r

# Run the financial life model with the input parameters
model_output = utils.financial_life_model(input_params)

# Display a plot
st.pyplot(utils.plot_model_output(model_output, variables_to_plot))
