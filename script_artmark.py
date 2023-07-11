import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import linregress
from scipy.stats import jarque_bera
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model

# Load data
df = pd.read_csv('ArtLichid2003-2018.csv')

# Step 1: Descriptive Analysis
print(df.describe())
# Step 2: Trend Analysis
plt.figure(figsize=(14, 8))
plt.plot(df['Date'], df['Value'])
xticks = np.arange(0, len(df['Date']), 10)
# plt.xticks(xticks, df['Date'][xticks])
# plt.title('Art Market Index Over Time')
# plt.xlabel('Time')
# plt.ylabel('Index')
# plt.show()


# Fit a (straight) trendline
# x = np.arange(len(df))
# slope, intercept, r_value, p_value, std_err = linregress(x, df['Value'])
# plt.plot(df['Date'], df['Value'], 'o', label='original data')
# plt.plot(df.index, intercept + slope*x, 'r', label='fitted line')
# plt.xticks(xticks, df['Date'][xticks])
# print(slope,"*x + ", intercept)
# plt.legend()
# plt.show()

# Fit a trendline (polynomial of degree 2)
# x = np.arange(len(df))
# z = np.polyfit(x, df['Value'], 2)
# p = np.poly1d(z)
# plt.plot(df['Date'], p(x), 'r--')
# plt.xticks(xticks, df['Date'][xticks])
# plt.show()
# equation = "y = %.6f * x^2 + %.6f * x + %.6f"%(p[0], p[1], p[2])
# print("Equation of the fitted line: ", equation)

# Step 3: Volatility Analysis - Rolling Standard Deviation
# df['Rolling Std Dev'] = df['Value'].rolling(window=12).std()
# plt.figure(figsize=(14, 8))
# plt.plot(df.index, df['Rolling Std Dev'])
# plt.title('Rolling Standard Deviation Over Time')
# plt.xlabel('Time')
# plt.xticks(xticks, df['Date'][xticks])
# plt.ylabel('Standard Deviation')
# plt.show()

#Garch model
# garch_model = arch_model(df['Value'], vol='Garch', p=1, q=1)
# garch_results = garch_model.fit()
# print(garch_results.summary())

# SARIMA model
sarima_model = SARIMAX(df['Value'], order=(2, 1, [3,6,9,12]), seasonal_order=(1, 1, 0, 12), trend = 'c')
sarima_results = sarima_model.fit(disp=-1)
std_resid = sarima_results.resid / sarima_results.resid.std()
#jb_test = jarque_bera(std_resid)
#jb_stat = jb_test[0]
#jb_pvalue = jb_test[1]
#print(jb_stat, jb_pvalue)
print(sarima_results.summary())

import matplotlib.pyplot as plt

# Plotting the residuals
plt.figure(figsize=(10, 6))
plt.plot(sarima_results.resid)
plt.xlabel('Observation')
plt.ylabel('Residual')
plt.title('Residuals Plot')
plt.xticks(xticks, df['Date'][xticks])
plt.show()