"""
Measuring the impact of pathologic features of penile squamous cell carcinomas in PD-L1 expression: A machine learning approach
@author: alcideschaux
"""

import numpy as np
import pandas as pd
import scipy.stats

# Loading the data
df = pd.read_pickle('PDL1.pkl')

"""
PATHOLOGIC FEATURES
"""
# Number of spots and variables
df.shape

# Number of pathology cases
df['sp'].nunique()

# Histologic subtype
df.groupby('subtype')['sp'].nunique().sort_values(ascending=False)

# Histologic grade
df['grade'].value_counts(sort=False)

# Host response
df['host_response'].value_counts(sort=False)

# Histologic grade and histologic subtype
tbl = df.groupby(['grade','subtype'])['n'].count().to_frame()
tbl_pivot = pd.pivot_table(tbl, index='subtype', columns='grade', values='n')
scipy.stats.chi2_contingency(tbl_pivot)

# Histologic subtype and host response
tbl = df.groupby(['subtype','host_response'])['n'].count().to_frame()
tbl_pivot = pd.pivot_table(tbl, index='subtype', columns='host_response', values='n')
scipy.stats.chi2_contingency(tbl_pivot)

# Histologic grade and host response
tbl = df.groupby(['grade','host_response'])['n'].count().to_frame()
tbl_pivot = pd.pivot_table(tbl, index='grade', columns='host_response', values='n')
scipy.stats.chi2_contingency(tbl_pivot)

"""
PD-L1
"""
# PD-L1 expression in tumor cells (percentages)
df['pdl1_tumor'].describe().round()

pdl1_pos = df['pdl1_tumor'].apply(lambda x: 'Positive' if x >= 1 else 'Negative')
pdl1_pos.value_counts(normalize=True)

df['pdl1_tumor_location'].value_counts(normalize=True)

# PD-L1 expression in tumor cells (H-scores)
df['pdl1_tumor_h'].describe().round()

# PD-L1 expression in intratumoral lymphocytes
df['pdl1_lymph'].describe().round()

# PD-L1 expression in tumor cells vs intratumoral lymphocytes
scipy.stats.spearmanr(df['pdl1_tumor'], df['pdl1_lymph'], nan_policy='omit')

# PD-L1 expression by histologic subtype
df.groupby('subtype')['pdl1_tumor'].describe().round()
scipy.stats.kruskal(*[dataframe['pdl1_tumor'].values for name, dataframe in df.groupby('subtype')], nan_policy='omit')

df.groupby('subtype')['pdl1_tumor_h'].describe().round()
scipy.stats.kruskal(*[dataframe['pdl1_tumor_h'].values for name, dataframe in df.groupby('subtype')], nan_policy='omit')

df.groupby('subtype')['pdl1_lymph'].describe().round()
scipy.stats.kruskal(*[dataframe['pdl1_lymph'].values for name, dataframe in df.groupby('subtype')], nan_policy='omit')

# PD-L1 expression by histologic grade
df.groupby('grade')['pdl1_tumor'].describe().round()
scipy.stats.kruskal(*[dataframe['pdl1_tumor'].values for name, dataframe in df.groupby('grade')], nan_policy='omit')

df.groupby('grade')['pdl1_tumor_h'].describe().round()
scipy.stats.kruskal(*[dataframe['pdl1_tumor_h'].values for name, dataframe in df.groupby('grade')], nan_policy='omit')

df.groupby('grade')['pdl1_lymph'].describe().round()
scipy.stats.kruskal(*[dataframe['pdl1_lymph'].values for name, dataframe in df.groupby('grade')], nan_policy='omit')

# PD-L1 expression by host response
df.groupby('host_response')['pdl1_tumor'].describe().round()
scipy.stats.kruskal(*[dataframe['pdl1_tumor'].values for name, dataframe in df.groupby('host_response')], nan_policy='omit')

df.groupby('host_response')['pdl1_tumor_h'].describe().round()
scipy.stats.kruskal(*[dataframe['pdl1_tumor_h'].values for name, dataframe in df.groupby('host_response')], nan_policy='omit')

df.groupby('host_response')['pdl1_lymph'].describe().round()
scipy.stats.kruskal(*[dataframe['pdl1_lymph'].values for name, dataframe in df.groupby('host_response')], nan_policy='omit')

# PD-L1 and impact of pathologic features
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import cross_val_score

def get_scores(target, features):
    y = df_ml.loc[:,target]
    X = df_ml.loc[:,features]
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), features)], remainder='passthrough')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = ct.fit_transform(X_train)
    X_test = ct.transform(X_test)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    var = explained_variance_score(y_test, y_pred)
    print('Variance explained: %g' %(var))

# Percentage of positive tumor cells
df_ml = df.loc[:,['subtype','grade','host_response','pdl1_tumor']].dropna()
target = 'pdl1_tumor'
get_scores(target, ['subtype'])
get_scores(target, ['grade'])
get_scores(target, ['host_response'])
get_scores(target, ['subtype','grade'])
get_scores(target, ['host_response','grade'])
get_scores(target, ['subtype','host_response'])
get_scores(target, ['subtype','grade','host_response'])

# H-scores in tumor cells
df_ml = df.loc[:,['subtype','grade','host_response','pdl1_tumor_h']].dropna()
target = 'pdl1_tumor_h'
get_scores(target, ['subtype'])
get_scores(target, ['grade'])
get_scores(target, ['host_response'])
get_scores(target, ['subtype','grade'])
get_scores(target, ['host_response','grade'])
get_scores(target, ['subtype','host_response'])
get_scores(target, ['subtype','grade','host_response'])

#Percentage of postive intratumoral lymphocytes
df_ml = df.loc[:,['subtype','grade','host_response','pdl1_lymph']].dropna()
target = 'pdl1_lymph'
get_scores(target, ['subtype'])
get_scores(target, ['grade'])
get_scores(target, ['host_response'])
get_scores(target, ['subtype','grade'])
get_scores(target, ['host_response','grade'])
get_scores(target, ['subtype','host_response'])
get_scores(target, ['subtype','grade','host_response'])
