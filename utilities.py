import numpy as np
import squigglepy as sq

# from squigglepy import K, M
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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


def plot_model_output(
    model_output,
    variables=None,
    alpha=0.1,
    mean_line_alpha=1,
    line_color="red",
    mean_line_width=2,
    background_color="black"
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
    fig, axs = plt.subplots(
        nrows=(len(model_output.keys()) + 1) // 2, ncols=2, figsize=(18, 12)
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
            color=line_color,
            alpha=mean_line_alpha,
            linewidth=mean_line_width,
        )  # plot mean path

        ax.plot(value, color=line_color, alpha=alpha)  # plot individual paths

        ax.set_title(key.replace("_", " ").title())  # prettify title
        ax.yaxis.set_major_formatter(formatter)  # format y axis with comma separator

        ax.set_facecolor(background_color) # set background color

    if len(model_output.keys()) % 2 != 0:
        fig.delaxes(axs[-1, -1])
    
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #    ax.label_outer()

    plt.tight_layout()
    # plt.show()
    return fig


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
    min_cash = input_params["min_cash"]
    min_market = input_params["min_market"]
    inflation_rate = input_params["inflation_rate"]
    ar_income_coefficients = input_params["ar_income_coefficients"]
    ar_income_sd = input_params["ar_income_sd"]
    ar_inflation_coefficients = input_params["ar_inflation_coefficients"]
    ar_inflation_sd = input_params["ar_inflation_sd"]
    r = input_params["r"]
    proportion_dissavings_from_cash = input_params["proportion_dissavings_from_cash"]
    beta = input_params["beta"]
    alpha = input_params["alpha"]

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

    # Generate income, inflation, and market returns paths
    income = gen_ar_income(
        years,
        ar_income_coefficients,
        life_cycle_income[0],
        ar_income_sd,
        life_cycle_income,
        min_income,
        m,
    )

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
    income = np.maximum(income / cumulative_inflation, min_income)
    market_returns = market_returns / cumulative_inflation

    for i in range(m):
        # Set initial conditions
        consumption[i, 0] = income[i, 0]
        savings[i, 0] = income[i, 0] - consumption[i, 0]
        cash[i, 0] = input_params["cash_start"] + savings[i, 0]
        market[i, 0] = input_params["market_start"] * (1 + market_returns[i, 0])
        non_financial_wealth[i, 0] = income[i, 0:].sum()
        financial_wealth[i, 0] = cash[i, 0] + market[i, 0]
        total_wealth[i, 0] = financial_wealth[i, 0] + non_financial_wealth[i, 0]

        for t in range(1, years):
            non_financial_wealth[i, t] = income[
                i, t:
            ].sum()  # Sum of future income is the non-financial wealth
            total_wealth[i, t] = (
                financial_wealth[i, t - 1] + non_financial_wealth[i, t]
            )  # Total wealth is sum of financial and non-financial wealth

            # Compute desired consumption, based on total wealth and MPC (Marginal Propensity to Consume)
            desired_consumption = income[i, t] + (1 - beta) * total_wealth[i, t]

            # Get total savings at time t
            total_savings = cash[i, t - 1] + market[i, t - 1]

            # If desired consumption is greater than total savings, adjust consumption
            if desired_consumption > total_savings:
                # Cannot consume more than total savings
                consumption[i, t] = total_savings
            else:
                consumption[i, t] = desired_consumption

            savings[i, t] = income[i, t] - consumption[i, t]

            # Compute dissavings, if any
            dissavings = max(0, consumption[i, t] - income[i, t])

            # Then dissave out of cash and market according to the real interest rate
            dissavings_from_cash = dissavings * proportion_dissavings_from_cash
            dissavings_from_market = dissavings - dissavings_from_cash

            # Compute cash and market wealth for time t after taking into account dissavings
            cash_wealth_after_dissavings = max(
                0, cash[i, t - 1] * (1 + r) - dissavings_from_cash
            )
            market_wealth_after_dissavings = max(
                0,
                market[i, t - 1] * (1 + market_returns[i, t]) - dissavings_from_market,
            )

            # If the dissavings from cash or market wealth are greater than the available cash or market wealth, adjust accordingly
            if dissavings_from_cash > cash[i, t - 1] * (1 + r):
                # If the dissavings from cash is greater than the available cash, dissave the remainder from the market wealth
                dissavings_from_market += dissavings_from_cash - cash[i, t - 1] * (
                    1 + r
                )
                dissavings_from_cash = cash[i, t - 1] * (1 + r)

            if dissavings_from_market > market[i, t - 1] * (1 + market_returns[i, t]):
                # If the dissavings from market wealth is greater than the available market wealth, dissave the remainder from the cash
                dissavings_from_cash += dissavings_from_market - market[i, t - 1] * (
                    1 + market_returns[i, t]
                )
                dissavings_from_market = market[i, t - 1] * (1 + market_returns[i, t])

            # Calculate cash and market wealth for time t after the adjustments
            cash[i, t] = cash_wealth_after_dissavings
            market[i, t] = market_wealth_after_dissavings

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
