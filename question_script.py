# Import necessary libraries
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Function to calculate Cronbach's alpha
def cronbach_alpha(df):
    # Number of columns
    N = df.shape[1]
    
    # Variance for every column
    variances = df.var(axis=0, ddof=1)
    
    # Total variance
    total_var = variances.sum()
    
    # Variance of sum of columns
    row_sums = df.sum(axis=1)
    total_var_sum = row_sums.var(ddof=1)
    
    # Calculate Cronbach's alpha
    alpha = (N / (N-1)) * (1 - total_var/total_var_sum)
    
    return alpha

# Load the Excel file
df = pd.read_excel('df_numeric.xlsx', engine='openpyxl')

# Replace missing values with 0
df.fillna(0, inplace=True)

# Drop string columns
#df = df.select_dtypes(exclude=['object'])
#filter relevant columns for cronbach's alpha
columns_cronbach = ['frequency_art', 'frequency_exhibition', 'buy_art',
       'Oil painting', 'Watercolor painting', 'Graphite drawing', 'Sculpture',
       'Photography', 'Generative art', 'Music', 'Video', 'NFT', 'Rare items',
       'Other', 'savings','romanian_origin', 'physical_digital',
       'art_investment', 'bought_art', 'Oil painting2', 'Watercolor painting2',
       'Graphite drawing2', 'Sculpture2', 'Photography2', 'Generative art2',
       'Music2', 'Video2', 'NFT2', 'Rare items2', 'Other2', 'bought_art2',
       'variety', 'quality', 'transparency']
df_cronbach = df[columns_cronbach]
# #output dataframe to excel
# df_cronbach.to_excel('df_numeric.xlsx')

#filter relevant columns for validity analysis
columns_validity = ['frequency_art', 'frequency_exhibition', 'buy_art','savings',
'art_investment', 'bought_art','bought_art2','variety', 'quality', 'transparency']
df_validity = df[columns_validity]
# add totals columns
totals = df_validity.sum(axis=1)
df_validity['totals'] = totals
#print(df)
#print(df.columns)
# Calculate and print Cronbach's alpha
# alpha = cronbach_alpha(df_cronbach)
# print(f"Cronbach's alpha: {alpha}")

#calculate correlation matrix
correlation_matrix = df_validity.corr()
# #output correlation matrix to excel
#correlation_matrix.to_csv('correlation_matrix.csv')

# # Calculate the p-values
# p_values = pd.DataFrame(index=correlation_matrix.columns, columns=correlation_matrix.columns)

# for i in correlation_matrix.columns:
    # for j in correlation_matrix.columns:
        # p_values[i][j] = pearsonr(df_validity[i], df_validity[j])[1]
# #output p_values matrix
# p_values.to_csv('p_values_matrix.csv')
# #print matrices
# print(correlation_matrix)
# print(p_values)

# #create correlation heatmap
# plt.figure(figsize=(10,10))
# sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f")
# plt.title("Heatmap of the Correlation Matrix")
# plt.savefig("heatmap_correlations.png")
# plt.close()

# #create p-values heatmap
# cmap_pvalues = LinearSegmentedColormap.from_list("", ['green','white'])
# plt.figure(figsize=(10,10))
# sns.heatmap(p_values.astype(float), cmap=cmap_pvalues, linewidths=.5, cbar=False)
# plt.title("Heatmap of the P-Values Matrix")
# plt.savefig("heatmap_pvalues.png")

# #t test difference of means art as an investment
# low_interest = df[df['art_investment'].isin([1, 2])]['buy_art']
# high_interest = df[df['art_investment'].isin([4,5])]['buy_art']

# t_stat, p_val = stats.ttest_ind(low_interest, high_interest)

# print(f'The t-statistic is: {t_stat}, the p-value is: {p_val}')

#t test physical vs digital
#gen pop
# t_stat, p_val = stats.ttest_1samp(df['physical_digital'], 3, alternative = 'greater')
# print(f'The t-statistic is: {t_stat}, the p-value is: {p_val}')
#gen z
# print(df.columns)
# t_stat, p_val = stats.ttest_1samp(df[df['age']=='18-24']['physical_digital'], 3)#, alternative = 'greater')
# print(f'The t-statistic is: {t_stat}, the p-value is: {p_val}')

# #x:(frequency, savings, bought_art), y:art
# X = df[['frequency_art', 'savings','bought_art']]
# y = df['buy_art']

# # Add a constant term to the independent variables
# X = sm.add_constant(X)

# # Fit the OLS regression model
# model = sm.OLS(y, X)
# results = model.fit()

# # Print the summary of the regression results
# print(results.summary())

#x:(romanian_origin,variety,quality,transparency), y:bought_art2
X = df[['romanian_origin', 'variety','quality', 'transparency']]
y = df['bought_art2']

# Add a constant term to the independent variables
X = sm.add_constant(X)

# Fit the OLS regression model
model = sm.OLS(y, X)
results = model.fit()

# Print the summary of the regression results
print(results.summary())