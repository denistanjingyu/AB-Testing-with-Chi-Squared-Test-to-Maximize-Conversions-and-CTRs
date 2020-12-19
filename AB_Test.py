# import libraries
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv('C:\\Users\\user\\Desktop\\ab_data.csv')

# data wrangling
# verify that control group users saw old page and treatment saw new page
df.groupby(['group', 'landing_page']).count()

# since unable to verify original group of users, let's remove them
df_cleaned = df.loc[(df['group'] == 'control') & (df['landing_page'] == 'old_page') |
                    (df['group'] == 'treatment') & (df['landing_page'] == 'new_page')]

# verify cleaned data
df_cleaned.groupby(['group', 'landing_page']).count()

# checking for duplicate values based on user id
# 1 duplicate found
df_cleaned['user_id'].duplicated().sum()

# locate user id for for duplicate value
# user saw new page twice and didn't convert both times
# for standardization, we will keep first trial for all users
df_cleaned[df_cleaned.duplicated(['user_id'], keep = False)]
df_cleaned = df.drop_duplicates(subset = 'user_id', keep = 'first')

# exploratory data analysis
# proportion of conversions for both groups
groups = df_cleaned.groupby(['group', 'landing_page', 'converted']).size()
groups.plot.bar()

# Re-arrrange data into 2x2 for Chi-Squared
# 1) Split groups into two separate DataFrames
a = df_cleaned[df_cleaned['group'] == 'control']
b = df_cleaned[df_cleaned['group'] == 'treatment']

# 2) A-click, A-noclick, B-click, B-noclick
a_click = a.converted.sum()
a_noclick = a.converted.size - a.converted.sum()
b_click = b.converted.sum()
b_noclick = b.converted.size - b.converted.sum()

# 3) Create np array
T = np.array([[a_click, a_noclick], [b_click, b_noclick]])

# Chi-Squared Test (chi-squared statistic and the p-value)
# p-value = 23%
# Assuming a 5% level of significance, we can deduce that the p-value is 
# greater than the alpha and that we do not reject the null hypothesis. 
# In simpler terms, there is no significance in conversions between the old 
# and new webpage.
print(scipy.stats.chi2_contingency(T, correction = False)[1])
