# %%

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('./data/nhanes_data_17_18.csv')
df.head()

sns.kdeplot( df['CarbohydrateGm_DR1TOT'] )
plt.show()

sns.ecdfplot(df['CarbohydrateGm_DR1TOT'])
plt.show()

x = df['CarbohydrateGm_DR1TOT'].dropna()
iqr = np.quantile( x, .75) - np.quantile( x, .25)

sns.boxplot( x = x )
plt.show()

# %%

def outlier_analyze(x):
    iqr = np.quantile( x, .75) - np.quantile( x, .25)
    uw = np.quantile( x, .75) + 1.5 * iqr # upper whisker
    lw = np.quantile( x, .25) - 1.5 * iqr # lower whisker
    upper_outlier = ( x > uw ).astype(int)
    lower_outlier = ( x < lw ).astype(int)
    outlier = upper_outlier + lower_outlier
    winsorize = ( upper_outlier * uw \
                 + lower_outlier * lw \
                 + (1-outlier) * x )
    return outlier, winsorize


outlier, winsorize = outlier_analyze(x)
# %%

sns.scatterplot( x=x, y=winsorize)
# %%

sns.violinplot(x=x,inner='quart',fill=False)
# %%

print('Work completed, things have changed!')