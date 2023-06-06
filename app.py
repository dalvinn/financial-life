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
    #initial_sidebar_state="expanded",
    #menu_items={
    #    'Get Help': 'https://www.extremelycoolapp.com/help',
    #    'Report a bug': "https://www.extremelycoolapp.com/bug",
    #    'About': "# This is a header. This is an *extremely* cool app!"
    #}
    )

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

st.title("Financial Life Model")

st.markdown(
    """
    This app simulates the possibilities for an individual's financial life.
    """
    )

st.markdown(
    """
    The repo for this app can be found [here](https://github.com/joel-becker/financial-life).
    If you have any questions or feedback, please contact me by [email](mailto:joelhbkr@gmail.com) or [twitter](https://twitter.com/joel_bkr).
    """
    )


with st.expander("How does the model work?"):
    st.markdown(
        """
        The model first simulates income, inflation, and market returns paths over time.
        It uses these paths to compute paths for consumption, savings, and wealth.
        """
        )


# Create a first row of sliders
st.markdown("### Parameters")

st.markdown("#### Background Financial Information")

row1_space1, row1_space2 = st.columns(2)

with row1_space1:
    income_start = st.slider(
        "Initial income", 20_000, 200_000, 40_000,
        help="This is the level of income you start with at the beginning of the simulation."
    )
    cash_start = st.slider(
        "Initial cash", 0, 100_000, 10_000, 
        help="This is the amount of cash you start with at the beginning of the simulation."
        )
    market_start = st.slider(
        "Initial market wealth", 0, 500_000, 10_000,
        help="This is the amount of market investments you start with at the beginning of the simulation."
    )

with row1_space2:
    years_until_retirement = st.slider(
        "Years until retirement", 0, 60, 45,
        help="This is the number of years until you retire."
    )
    min_income = st.slider(
        "Minimum income",
        0,
        50_000,
        15_000,
        help="The reservation income, a lower bound that you don't expect to dip below.",
    )
    retirement_income = st.slider(
        "Retirement income", 0, 100_000, 10_000,
        help="This is the annual income you expect to earn during retirement."
    )

st.markdown("#### Spending possibilities")

row2_space1, row2_space2 = st.columns(2)

with row2_space1:
    income_fraction_consumed = st.slider(
        "Fraction of annual income consumed (before retirement)", 0.0, 1.0, 0.4,
        help="This is only one component of consumption: the other is the fraction of your annualized total wealth."
        )
    wealth_fraction_consumed_before_retirement = st.slider(
        "Fraction of annualized total wealth consumed (before retirement)", 0.0, 1.0, 0.35,
        help="Annualized total wealth refers to your total wealth (including all future income, excluding income this year) divided by the number of years you have left to live. This is only one component of consumption: the other is the fraction of your annual income."
        )

with row2_space2:
    wealth_fraction_consumed_after_retirement = st.slider(
        "Fraction of annualized total wealth consumed (after retirement)", 0.0, 1.0, 0.45,
        help = "Annualized total wealth refers to your total wealth (including all future income, excluding income this year) divided by the number of years you have left to live. Beyond this, it is assumed that you will spend all of your retirement income."
        )

st.markdown("#### Hidden Settings")

# Create a collapsible section for the second row
row3_space1, row3_space2, row3_space3 = st.columns(3)

with row3_space1:
    with st.expander("Advanced Settings"):
        max_cash_threshold = st.slider(
            "Maximum cash on hand", 3_000, 30_000, 5_000,
            help="The maximum amount of cash you want to hold at any given time. If cash is above this amount at the beginning of a period, the model will transfer the resources to market investments."
        )
        min_cash_threshold = st.slider(
            "Minimum cash on hand", 0, 10_000, 5_000,
            help="The minimum amount of cash you want to hold at any given time. If cash is below this amount at the beginning of a period, the model will transfer the resources from market investments."
        )

with row3_space2:
    with st.expander("Plot Settings"):
        m = st.slider(
            "Number of simulated paths", 100, 1_000, 200
        )
        years = st.slider(
            "Number of years left to live", 10, 80, 70
        )

with row3_space3:
    with st.expander("Economic Conditions"):
        r = st.slider(
            "Interest rate", 0.0, 0.1, 0.02, help="The rate of return on savings."
        )
        inflation_rate = st.slider(
            "Inflation rate", 0.0, 0.1, 0.02
        )

st.markdown("### Results")

variables_to_plot = st.multiselect(
    "Select variables to plot",
    options=[
        "income",
        "savings",
        "consumption",
        "financial_wealth",
        "cash",
        "market",
    ],
    default=["income", "savings", "consumption", "financial_wealth"],
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
    income_start + (year * 2 * 1000) for year in range(input_params["years"])
]
input_params["min_income"] = min_income
input_params["inflation_rate"] = inflation_rate
input_params["r"] = r
input_params["years_until_retirement"] = years_until_retirement
input_params["retirement_income"] = retirement_income

# Run the financial life model with the input parameters
model_output = utils.financial_life_model(input_params)

# Display a plot
st.pyplot(utils.plot_model_output(model_output, variables_to_plot))

with st.expander("How do I read these plots?"):
    st.markdown(
        """
        The plots each show the simulated paths of a selected variables over time. 
        Here's what each variable represents:
        - Income: Your income each year.
        - Savings: The amount you save (or dissave) each year.
        - Consumption: The amount you consume each year.
        - Financial Wealth: Your total wealth, including cash and market investments.
        - Cash: Your cash savings each year.
        - Market: Your market investments each year.
        """
        )
