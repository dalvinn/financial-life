import sys

sys.path.append("src")

import utilities as utils
import parameters as params

model_output = utils.financial_life_model(params.input_params)

utils.plot_model_output(
    model_output, variables=["income", "cash", "market", "consumption"], alpha=0.01
)
