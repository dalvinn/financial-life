import toml

# Read the config file
config = toml.load(".streamlit/config.toml")

# Extract the color configuration
primary_color = config["theme"]["primaryColor"].strip()
background_color = config["theme"]["backgroundColor"].strip()
secondary_background_color = config["theme"]["secondaryBackgroundColor"].strip()
text_color = config["theme"]["textColor"].strip()

# Define matplotlib style settings
mplstyle = {
    "axes.facecolor": background_color,
    "figure.facecolor": background_color,
    "text.color": text_color,
    "axes.edgecolor": primary_color,
    "axes.labelcolor": primary_color,
    "xtick.color": primary_color,
    "ytick.color": primary_color,
    "figure.autolayout": True,

    'axes.grid': True,
    'grid.color': primary_color,
    'grid.alpha': 0,

    'lines.color': primary_color,
    'patch.facecolor': primary_color,
    'patch.edgecolor': primary_color,
}

# Write matplotlib style settings to .mplstyle file
#with open("plot_style.mplstyle", "w") as f:
#    for key, value in mplstyle.items():
#        print(f"{key}:{value}")
#        f.write(f"{key}:{value}\n")
#
#with open("plot_style.mplstyle", "w") as f:
#    for key, value in mplstyle.items():
#        # Use repr() function to get a string containing a printable representation of 'value'
#        print(f"{key}: {repr(value)}")
#        f.write(f"{key}: {repr(value)}\n")
