import pandas as pd
import numpy as np

# Open dataset
df = pd.read_csv('PFCK_PRY_DF.csv')

# Add column for ID each case
df['n'] = range(len(df))

# Rename and order levels in categorical variables
df = df.astype({'grade':'category','host_response':'category'})
df['grade'].cat.set_categories(['Grade 1','Grade 2','Grade 3'], ordered=True, inplace=True)
df['host_response'].cat.rename_categories({'No inflammatory cells':'No','Rare inflammatory cells':'Mild','Intense inflammation':'Intense','Lymphoid aggregates':'Moderate'}, inplace=True)

df.info()

# List of pathologic features
pathologic = ['n','sp','subtype','grade','host_response']

# List of PD-L1 features
pdl1_features = ['pdl1_tumor','pdl1_tumor_location','pdl1_tumor_h','pdl1_lymph']

# List of CD8-Ki67 features
cd8ki67_features = ['cd8_tumor','cd8ki67_tumor','cd8_stroma','cd8ki67_stroma']

# Final dataset for the analysis
features = pathologic + pdl1_features + cd8ki67_features
data = df.loc[:,features]

data.info()

data.to_pickle('data.pkl')
data.to_json('data.json')
