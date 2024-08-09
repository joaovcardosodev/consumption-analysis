import pandas as pd
from datetime import date, datetime
import dateutil.parser
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.deterministic import DeterministicProcess


consumo = pd.read_excel("C:/Users/U356242/Desktop/Python/CONSUMO TRAFO 30KVA.xlsx", parse_dates=['Entrado em'])
consumo = consumo.set_index('Entrado em')

# for i in consumo.index:
#     data = str(consumo['Entrado em'][i])
#     datetime_object = dateutil.parser.parse(data)
#     data = datetime_object.strftime("%d/%m/%y")
#     data = datetime.strptime(data, '%d/%m/%y')
#     consumo['Entrado em'][i] = data

consumo_group = consumo.groupby(by='Entrado em').sum()

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 5))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)

# moving_average = consumo_group.rolling(
#     window=365,       # 365-day window
#     center=True,      # puts the average at the center of the window
#     min_periods=183,  # choose about half the window size
# ).mean()              # compute the mean (could also do median, std, min, max, ...)

# ax = consumo_group.plot(style=".", color="0.5")
# moving_average.plot(
#     ax=ax, linewidth=3, title="Consumo - 365-Day Moving Average", legend=False,
# );

dp = DeterministicProcess(
    index=consumo_group.index,  # dates from the training data
    constant=True,       # dummy feature for the bias (y_intercept)
    order=1,             # the time dummy (trend)
    drop=True,           # drop terms if necessary to avoid collinearity
)
# `in_sample` creates features for the dates given in the `index` argument
X = dp.in_sample()

print(X.head())