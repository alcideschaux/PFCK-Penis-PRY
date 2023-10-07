""" FUNCTIONS FOR EXPLORATORY DATA ANALYSIS """

'''
Tables
'''
# Defining a function to create a pivot table
def create_pivot(rows, columns):
    tbl = df.groupby([columns, rows])['n'].count().astype('int').to_frame()
    tbl_pivot = pd.pivot_table(tbl, index=rows, columns=columns, values='n').fillna(0).astype('int')
    return tbl_pivot

# Defining a function to describe numeric variables
def create_descriptive(x):
    return df[x].describe().round().to_frame()

# Defining a function to provide grouped descritive statistics
def create_descriptives(x, group):
    return df.groupby(group)[x].describe().round()

'''
Plots
'''
# Barplot
def plot_bar(x, y):
    ax = sns.countplot(data=df, x=x, hue=y)

# Distribution plot
def plot_kde(x, bins=10):
    sns.displot(df[x].dropna(), kde=False, bins=bins)
    plt.show()

# Boxenplot
def plot_box(x, group):
    ax = sns.boxenplot(data=df, x=group, y=x)

# Pointplot
def plot_point(x, group):
    ax = sns.pointplot(x=group, y=x, ci=None, estimator=np.median, data=df)
    
# Pairplot
def plot_pair(vars):
    df_corr = df.loc[:,vars]
    df_corr.dropna(inplace=True)
    ax = sns.pairplot(df_corr)
    
'''
Tests
'''
# Chi-squared test
def do_chi2(rows, columns):
    tbl_pivot = create_pivot(rows, columns)
    chi2, p, dof, expected = scipy.stats.chi2_contingency(tbl_pivot)
    print('Chi2 statistic: %g\nDegrees of freedom: %g\nP value: %g' %(chi2, dof, p))

# Kruskal-Wallis test
def do_kw(x, group):
    stat, p = scipy.stats.kruskal(*[data[x].values for name, data in df.groupby(group)], nan_policy='omit')
    print('Kruskal-Wallis statistic: %g\nP value: %g' %(stat, p))

# Spearman correlation test, rho coefficients
def do_spearman_rho(vars):
    df_corr = df.loc[:,vars]
    df_corr.dropna(inplace=True)
    x = np.array(df_corr)
    x_cols = df_corr.columns
    rho, p = scipy.stats.spearmanr(x)
    return pd.DataFrame(rho, index=x_cols, columns=x_cols).round(2)

# Spearman correlation test, p values
def do_spearman_p(vars):
    df_corr = df.loc[:,vars]
    df_corr.dropna(inplace=True)
    x = np.array(df_corr)
    x_cols = df_corr.columns
    rho, p = scipy.stats.spearmanr(x)
    return pd.DataFrame(p, index=x_cols, columns=x_cols)


""" FUNCTIONS FOR MODEL SELECTION """

# To create dummy variables for pathologic features
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Models that will be evaluated
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# To perform cross-validation
from sklearn.model_selection import cross_validate

# Metrics to be used for evaluating model performance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function to estimate model performance

def model_performance(target, regressor):
    # Creating the dataset
    features = ['subtype','grade','host_response']
    target = [target]
    data = df.loc[:,features+target].dropna().reset_index()
    
    # Selecting target and features
    X = data.loc[:,features]
    y = data.loc[:,target]
    y = np.ravel(y)

    # Creating dummy variables for pathologic features (which are all categorical variables)
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), features)], remainder='passthrough')
    X = ct.fit_transform(X)

    # Defining the regressor
    regressor = regressor

    # Estimating metrics from cross-validation
    results = cross_validate(regressor, X, y, cv=10, scoring=('neg_mean_absolute_error','neg_mean_squared_error','r2'))

    # Printing metrics for the regressor
    print('MAE: %g' %np.mean(results['test_neg_mean_absolute_error']))
    print('MSE: %g' %np.mean(results['test_neg_mean_squared_error']))
    print('R2: %g' %np.mean(results['test_r2']))
    

""" FUNCTION FOR EVALUATING IMPACT ON PATHOLOGIC FEATURES """

from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score

# Function to estimate the explained variance score
def get_scores(target, features, regressor):
    y = df_ml.loc[:,target]
    X = df_ml.loc[:,features]
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), features)], remainder='passthrough')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = ct.fit_transform(X_train)
    X_test = ct.transform(X_test)
    regressor = regressor
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    var = explained_variance_score(y_test, y_pred)
    print('Variance explained by %s: %g' %(str(features), var*100))
    
# Function to return explained variance scores for pathologic features
def get_all_scores(target, regressor):
    features = ['subtype','grade','host_response',target]
    global df_ml
    df_ml = df.loc[:,features].dropna()
    get_scores(target, ['subtype'], regressor)
    get_scores(target, ['grade'], regressor)
    get_scores(target, ['host_response'], regressor)
    get_scores(target, ['subtype','grade'], regressor)
    get_scores(target, ['host_response','grade'], regressor)
    get_scores(target, ['subtype','host_response'], regressor)
    get_scores(target, ['subtype','grade','host_response'], regressor)