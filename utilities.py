import numpy as np
import squigglepy as sq

# from squigglepy import K, M
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import create_style_script as css

# Use custom plot style
plt.rcParams.update(css.mplstyle)
#plt.style.use('plot_style.mplstyle')


def gen_market_returns(n, mean, sd, m):
    """
    Generate the market returns for each year.

    Parameters:
    n (int): The number of years.
    mean (float): The mean market return.
    sd (float): The standard deviation of market returns.

    Returns:
    ndarray: The market returns for each year.
    """
    return [sq.norm(mean=mean, sd=sd) @ n for m in range(m)]


def gen_ar_income(n, ar_coefficients, start, sd, baseline, min_income, m):
    """
    Generate m paths of n years of income following an AR(p) process with deviations from a lifecycle baseline.

    Parameters:
    n (int): The number of years.
    ar_coefficients (list): The coefficients of the AR process.
    start (float): The starting income.
    sd (float): The standard deviation of the random shocks.
    baseline (array): The baseline income for each year.
    min_income (float): The minimum income.
    m (int): The number of paths.

    Returns:
    ndarray: An m x n array where each row is a different path.
    """
    p = len(ar_coefficients)
    income = np.full((m, n), start)  # initialize income array
    shocks = np.zeros((m, n))  # initialize shocks array
    shocks[:, :p] = np.random.normal(0, sd, (m, p))  # initialize first p shocks

    for i in range(m):
        for t in range(p, n):
            # Calculate the new shock value based on the AR process
            new_shock = np.dot(
                ar_coefficients, shocks[i, t - p : t][::-1]
            ) + np.random.normal(0, sd)
            shocks[i, t] = new_shock  # update the shock for path i at year t

            # Compute the income for year t
            income[i, t] = max(baseline[t] + new_shock, min_income)

    return income


def gen_combined_income(n, ar_coefficients, start, sd, baseline, min_income, m, years_until_retirement, retirement_income, interest_rate_cumulative):
    """
    Generate m paths of n years of income, combining pre-retirement and post-retirement income.

    Parameters:
    n (int): The number of years.
    ar_coefficients (list): The coefficients of the AR process.
    start (float): The starting income.
    sd (float): The standard deviation of the random shocks.
    baseline (array): The baseline income for each year.
    min_income (float): The minimum income.
    m (int): The number of paths.
    years_until_retirement (int): The number of years until retirement.
    retirement_income (float): The retirement income.
    interest_rate_cumulative (ndarray): The cumulative interest rate for each year.

    Returns:
    ndarray: An m x n array where each row is a different path.
    """
    # Generate pre-retirement income
    pre_retirement_income = gen_ar_income(
        years_until_retirement,
        ar_coefficients,
        start,
        sd,
        baseline[:years_until_retirement],
        min_income,
        m,
    )

    # Generate post-retirement income
    if n > years_until_retirement:
        post_retirement_income = np.zeros((m, n - years_until_retirement))
        for t in range(n - years_until_retirement):
            post_retirement_income[:, t] = retirement_income * interest_rate_cumulative[:, years_until_retirement + t]

        # Combine pre-retirement and post-retirement income
        combined_income = np.hstack((pre_retirement_income, post_retirement_income))
    else:
        combined_income = pre_retirement_income

    return combined_income



def gen_ar_inflation(n, ar_coefficients, sd, inflation_rate, m):
    """
    Generate m paths of n years of inflation following an AR(p) process with deviations from an inflation rate baseline.

    Parameters:
    n (int): The number of years.
    ar_coefficients (list): The coefficients of the AR process.
    sd (float): The standard deviation of the random shocks.
    inflation_rate (float): The baseline inflation rate for each year.
    m (int): The number of paths.

    Returns:
    ndarray: An m x n array where each row is a different path.
    """
    p = len(ar_coefficients)
    inflation = np.full((m, n), inflation_rate)  # initialize inflation array
    shocks = np.zeros((m, n))  # initialize shocks array
    shocks[:, :p] = np.random.normal(0, sd, (m, p))  # initialize first p shocks

    for i in range(m):
        for t in range(p, n):
            # Calculate the new shock value based on the AR process
            new_shock = np.dot(
                ar_coefficients, shocks[i, t - p : t][::-1]
            ) + np.random.normal(0, sd)
            shocks[i, t] = new_shock  # update the shock for path i at year t

            # Compute the inflation for year t
            inflation[i, t] = inflation_rate + new_shock

    return inflation


def financial_life_model(input_params):
    """
    Simulate the financial life of an individual over a certain number of years.

    Parameters:
    input_params (tuple): A tuple containing the following inputs:
        m (int): The number of paths to simulate.
        years (int): The number of years to simulate.
        cash_start (float): The initial cash.
        market_start (float): The initial market wealth.
        income_start (float): The initial income.
        life_cycle_income (float): The baseline lifecycle income.
        min_income (float): The minimum level of income.
        min_cash (float): The minimum level of cash.
        min_market (float): The minimum level of market wealth.
        inflation_rate (float): The inflation rate.
        ar_income_coefficients (list): The coefficients of the AR process for income.
        ar_income_sd (float): The standard deviation of the AR process for income.
        ar_inflation_coefficients (list): The coefficients of the AR process for inflation.
        ar_inflation_sd (float): The standard deviation of the AR process for inflation.
        r (float): The constant risk-free interest rate.
        proportion_dissavings_from_cash (float): The proportion of dissavings that comes from cash.
        beta (float): The discount factor.
        alpha (float): The coefficient of relative risk aversion.

    Returns:
    dict: A dictionary containing the simulated paths of income, inflation, cash, market, wealth, consumption, savings and non_financial_wealth.
    """
    # Unpack parameters
    m = input_params["m"]
    years = input_params["years"]
    # income_start = input_params["income_start"]
    life_cycle_income = input_params["life_cycle_income"]
    min_income = input_params["min_income"]
    inflation_rate = input_params["inflation_rate"]
    ar_income_coefficients = input_params["ar_income_coefficients"]
    ar_income_sd = input_params["ar_income_sd"]
    ar_inflation_coefficients = input_params["ar_inflation_coefficients"]
    ar_inflation_sd = input_params["ar_inflation_sd"]
    r = input_params["r"]
    years_until_retirement = input_params["years_until_retirement"]
    years_until_death = input_params["years_until_death"]
    retirement_income = input_params["retirement_income"]
    wealth_fraction_consumed_after_retirement = input_params["wealth_fraction_consumed_after_retirement"]

    # Preallocate arrays
    income = np.zeros((m, years))
    inflation = np.zeros((m, years))
    cash = np.zeros((m, years))
    market = np.zeros((m, years))
    financial_wealth = cash + market
    consumption = np.zeros((m, years))
    savings = np.zeros((m, years))
    non_financial_wealth = np.zeros((m, years))
    total_wealth = financial_wealth + non_financial_wealth

    inflation = gen_ar_inflation(
        years,
        ar_inflation_coefficients,
        ar_inflation_sd,
        inflation_rate,
        m,
    )

    market_returns = gen_market_returns(years, 0.05, 0.15, m)

    # Derived parameters
    # marginal_propensity_to_consume = r / (1 + r)
    cumulative_inflation = np.cumprod(1 + inflation, axis=1)
    real_interest_rate = r - inflation
    interest_rate_cumulative = np.cumprod(1 + (r - np.zeros((m, years))), axis=1)
    market_returns = market_returns / cumulative_inflation

    # Generate income, inflation, and market returns paths
    income = gen_combined_income(
        years,
        ar_income_coefficients,
        life_cycle_income[0],
        ar_income_sd,
        life_cycle_income,
        min_income,
        m,
        years_until_retirement,
        retirement_income,
        interest_rate_cumulative
    )

    income = np.maximum(income, min_income) / cumulative_inflation

    for i in range(m):
        # Set initial conditions
        non_financial_wealth[i, 0] = income[i, 0:].sum()
        financial_wealth[i, 0] = input_params["cash_start"] + input_params["market_start"]
        total_wealth[i, 0] = financial_wealth[i, 0] + non_financial_wealth[i, 0]
        remaining_total_wealth_annualized = (total_wealth[i, 0] - income[i, 0]) / years

        consumption[i, 0] = (
            input_params["income_fraction_consumed"] * income[i, 0] 
            + input_params["wealth_fraction_consumed_before_retirement"] * remaining_total_wealth_annualized
        )
        savings[i, 0] = income[i, 0] - consumption[i, 0]
        cash[i, 0] = input_params["cash_start"] + savings[i, 0]
        market[i, 0] = input_params["market_start"] * (1 + market_returns[i, 0])
        non_financial_wealth[i, 0] = income[i, 0:].sum()
        financial_wealth[i, 0] = cash[i, 0] + market[i, 0]
        total_wealth[i, 0] = financial_wealth[i, 0] + non_financial_wealth[i, 0]

        for t in range(1, years):
            # Adjust cash and market wealth before considering income and consumption
            if cash[i, t - 1] < input_params["min_cash_threshold"]:
                transfer_to_cash = min(input_params["min_cash_threshold"] - cash[i, t - 1], market[i, t - 1])
                cash[i, t - 1] += transfer_to_cash
                market[i, t - 1] -= transfer_to_cash
            elif cash[i, t - 1] > input_params["max_cash_threshold"]:
                transfer_to_market = cash[i, t - 1] - input_params["max_cash_threshold"]
                cash[i, t - 1] -= transfer_to_market
                market[i, t - 1] += transfer_to_market

            non_financial_wealth[i, t] = income[
                i, t:
            ].sum()  # Sum of future income is the non-financial wealth
            total_wealth[i, t] = (
                financial_wealth[i, t - 1] + non_financial_wealth[i, t]
            )  # Total wealth is sum of financial and non-financial wealth

            remaining_total_wealth_annualized = (total_wealth[i, t - 1] - income[i, t]) / (years - t)

            # Compute desired consumption
            if t < years_until_retirement:
                consumption_from_income = input_params["income_fraction_consumed"] * income[i, t]
                consumption_from_wealth = input_params["wealth_fraction_consumed_before_retirement"] * remaining_total_wealth_annualized
            else:
                consumption_from_income = retirement_income
                consumption_from_wealth = wealth_fraction_consumed_after_retirement * remaining_total_wealth_annualized

            desired_consumption = consumption_from_income + consumption_from_wealth

            # Get total savings at time t
            total_savings = cash[i, t - 1] + market[i, t - 1]

            # If desired consumption is greater than total savings, adjust consumption
            if desired_consumption > total_savings:
                # Cannot consume more than total savings
                consumption[i, t] = total_savings
            else:
                consumption[i, t] = desired_consumption

            # Compute savings and allocate them to cash and market
            #if t < years_until_retirement:
            savings[i, t] = income[i, t] - consumption[i, t]
            if cash[i, t - 1] < input_params["max_cash_threshold"]:
                savings_to_cash = min(savings[i, t], input_params["max_cash_threshold"] - cash[i, t - 1])
                savings_to_market = max(0, savings[i, t] - savings_to_cash)
            else:
                savings_to_cash = 0
                savings_to_market = savings[i, t]
            #else:
            #    savings[i, t] = 0

            # Update cash and market wealth
            cash[i, t] = cash[i, t - 1] * (1 + r) + savings_to_cash
            market[i, t] = market[i, t - 1] * (1 + market_returns[i, t]) + savings_to_market

            # Compute dissavings and adjust cash and market wealth
            dissavings = max(0, - savings[i, t])
            if cash[i, t] > input_params["min_cash_threshold"]:
                dissavings_from_cash = min(dissavings, cash[i, t] - input_params["min_cash_threshold"])
                dissavings_from_market = max(0, dissavings - dissavings_from_cash)
            else:
                dissavings_from_cash = 0
                dissavings_from_market = dissavings

            cash[i, t] = max(0, cash[i, t] - dissavings_from_cash)
            market[i, t] = max(0, market[i, t] - dissavings_from_market)

            # Compute financial wealth
            financial_wealth[i, t] = cash[i, t] + market[i, t]

    model_output = {
        "income": income,
        "inflation": inflation,
        "cash": cash,
        "market": market,
        "financial_wealth": financial_wealth,
        "consumption": consumption,
        "savings": savings,
        "non_financial_wealth": non_financial_wealth,
        "total_wealth": total_wealth,
    }

    return model_output


def plot_model_output(
    model_output,
    variables=None,
    alpha=0.03,
    mean_line_alpha=1,
    #line_color="red",
    mean_line_width=2,
    #background_color="black"
):
    """
    This function plots the output of the financial life model.

    Parameters:
    - model_output: A dictionary where keys are labels and values are np.array of time series data.
    - variables: A list of variables to plot. If None, all variables are plotted. Default is None.
    - alpha: Transparency for individual paths. Default is 0.2.
    - mean_line_alpha: Transparency for mean path. Default is 1 (no transparency).
    - mean_line_color: Color for mean path. Default is 'red'.
    - mean_line_width: Line width for mean path. Default is 2.

    Returns:
    - Nothing, but it shows a matplotlib plot.
    """
    # Filter variables
    if variables is not None:
        model_output = {k: v for k, v in model_output.items() if k in variables}

    # Create a figure and a set of subplots
    nrows = (len(model_output.keys()) + 1) // 2

    if len(model_output.keys()) > 1:
        ncols = 2
    else:
        ncols = 1

    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(6 * ncols, 3 * nrows)
    )

    # This will remove top/right box/border around the subplots
    for ax in axs.flatten():
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    # Format y axis as currency
    formatter = ticker.FormatStrFormatter("$%1.0f")
    formatter = ticker.FuncFormatter(lambda x, pos: "${:,.0f}".format(x))

    # Flatten the axs array and iterate over it and the items in the dictionary at the same time
    for ax, (key, value) in zip(axs.flatten(), model_output.items()):
        # Transpose the data
        value = np.transpose(value)

        ax.plot(
            value.mean(axis=1),
            color=css.primary_color,
            alpha=mean_line_alpha,
            linewidth=mean_line_width,
        )  # plot mean path

        ax.plot(
            value, 
            color=css.primary_color, 
            alpha=alpha
            )  # plot individual paths

        ax.set_title(key.replace("_", " ").title())  # prettify title
        ax.yaxis.set_major_formatter(formatter)  # format y axis with comma separator

        #ax.set_facecolor(background_color) # set background color
        if np.min(value) >= 0:
            ax.set_ylim(bottom=0)

    if len(model_output.keys()) % 2 != 0:
        fig.delaxes(axs[-1, -1])

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #    ax.label_outer()

    for ax in axs[-1, :]:
        ax.set_xlabel("Years from present")
    
    for ax in axs[:, -0]:
        ax.set_ylabel("2023 USD")
        
    plt.tight_layout()
    # plt.show()
    return fig

