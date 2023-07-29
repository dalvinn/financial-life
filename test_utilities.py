import utilities
import parameters as params

def test_financial_life_model():
    # Create a parameters dictionary
    test_params = params.input_params
    test_params["m"] = 300
    test_params["years"] = test_params["years_until_retirement"] + 10
    test_params["cash_start"] = 10000
    test_params["market_start"] = 30000
    test_params["life_cycle_income"] = [
        60000 + (year * 3 * 1000) for year in range(test_params["years"])
    ]
    test_params["min_income"] = 10000
    test_params["inflation_rate"] = 0.02
    test_params["r"] = 0.02

    # Run your model
    model_output = utilities.financial_life_model(test_params)

    # Assert something about the output
    assert len(model_output) > 0, "Model output is empty"

