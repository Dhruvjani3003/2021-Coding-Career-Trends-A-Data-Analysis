# ==== Import Libraries ====
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

# ==== Load Dataset ====
df = pd.read_csv('2021_New_Coder_Survey.csv', low_memory=False)

# ==== Data Cleaning ====
df.drop_duplicates(inplace=True)

def rename_t_c(df, column_name, short_name):
    df.rename(columns={column_name: short_name}, inplace=True)
    return df

df = rename_t_c(df, '22. About how much money did you earn last year from any job or employment (in US Dollars)? ', 'data_of_earning')
df = rename_t_c(df, '7. About how many hours do you spend learning each week?', '7_hours_s_l')
df = rename_t_c(df, '23. How old are you?', 'col23')
df = rename_t_c(df, '41. Before you got your last job, how many months did you spend looking for a job?', 'col41')
df = rename_t_c(df, '42. If you are working, thinking about the next 12 months, how likely do you think it is that you will lose your job or be laid off? ', 'col42')
df = rename_t_c(df, '43. If you are working, how easy would it be for you to find a job with another employer with approximately the same income and fringe benefits you now have? ', 'col43')
df = rename_t_c(df, '8. About how many months have you been programming?', 'col8')

# ==== Exploratory Data Analysis (EDA) ====
plt.figure(figsize=(10, 6))
sns.barplot(x=df['data_of_earning'].value_counts().sort_index().index,
            y=df['data_of_earning'].value_counts().sort_index().values,
            hue=df['data_of_earning'].value_counts().sort_index().index, legend=False)
plt.title('Earnings Last Year')
plt.xlabel('Amount Earned (in US Dollars)')
plt.ylabel('Number of Respondents')
plt.xticks(rotation=45)
plt.show()

sns.scatterplot(x='col23', y='data_of_earning', data=df)
plt.title('Income vs. Age')
plt.show()

sns.scatterplot(x='7_hours_s_l', y='data_of_earning', data=df)
plt.title('Income vs. Hours Spent Learning')
plt.show()

    # ==== Outlier Removal ====
numerical_cols = ['7_hours_s_l', 'col23', 'col41', 'col42', 'col43']
for col in numerical_cols:
    valid_indices = df[col].notna()
    z_scores = np.abs(stats.zscore(df.loc[valid_indices, col]))
    # Create mask only for valid indices and expand to full DataFrame
    temp_mask = pd.Series(False, index=df.index)
    temp_mask.loc[valid_indices] = z_scores < 3
    df = df.loc[temp_mask]

# ==== Histograms & Boxplots ====
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
for i, attribute in enumerate(['7_hours_s_l', 'col23']):
    sns.histplot(df[attribute].dropna(), ax=axes[i, 0], kde=True)
    sns.boxplot(x=df[attribute].dropna(), ax=axes[i, 1])
plt.tight_layout()
plt.show()

# ==== Scatter Matrix ====
df_filtered = df.dropna(subset=['col8', 'data_of_earning', 'col23'])
scatter_matrix(df_filtered[['col8', 'data_of_earning', 'col23']], figsize=(12, 8), alpha=0.5)
plt.show()

# ==== Data Transformation & Normalization ====
pt = PowerTransformer(method='yeo-johnson')
transformed_data = pt.fit_transform(df[numerical_cols].dropna())
df_transformed = pd.DataFrame(transformed_data, columns=numerical_cols, index=df[numerical_cols].dropna().index)

scaler = StandardScaler()
normalized_data = scaler.fit_transform(df_transformed)
df_normalized = pd.DataFrame(normalized_data, columns=numerical_cols, index=df_transformed.index)

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.hist(df['col23'].dropna(), bins=20, alpha=0.5, label='Original')
plt.hist(df_transformed['col23'], bins=20, alpha=0.5, label='Transformed')
plt.legend()
plt.title('Histogram of Age')

plt.subplot(1, 2, 2)
plt.scatter(df['col23'].dropna(), df['7_hours_s_l'].dropna(), alpha=0.5, label='Original')
plt.scatter(df_transformed['col23'], df_transformed['7_hours_s_l'], alpha=0.5, label='Transformed')
plt.legend()
plt.title('Scatter of Age & Hours Learning')
plt.show()

# ==== Clustering (KMeans) ====
X = SimpleImputer(strategy='mean').fit_transform(df[['col42', 'col23']].dropna())
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

plt.figure(figsize=(10, 6))
for i in range(3):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], label=f'Cluster {i+1}')
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='x', label='Centroids')
plt.legend()
plt.title('K-Means Clustering')
plt.show()

# ==== Classification (Decision Tree) ====
df = df[~df['data_of_earning'].isin(["I don’t know", "I don't want to answer", "None"])]
def categorize_income(income_range):
    if pd.isna(income_range):
        return None
    elif income_range == 'Under $1,000':
        return 0
    elif income_range == '$250,000 or over':
        return 1
    else:
        try:
            lower = int(income_range.split(' to ')[0].replace('$', '').replace(',', ''))
            return 1 if lower >= 30000 else 0
        except:
            return None

df['high_income'] = df['data_of_earning'].apply(categorize_income)
df = df.dropna(subset=['high_income'])

X = df[['col23']].dropna()
y = df['high_income'].dropna()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print("Decision Tree Accuracy:", accuracy_score(y_test, model.predict(X_test)))
print(classification_report(y_test, model.predict(X_test)))

# ==== Classification (Logistic Regression) ====
non_numeric_cols = ['col42', 'data_of_earning']
data_numeric = df.drop(columns=[col for col in df.columns if df[col].dtype == 'object' and col not in non_numeric_cols], errors='ignore')
X = data_numeric.drop(columns=['col42'], errors='ignore')
y = data_numeric['col42']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = SimpleImputer().fit_transform(X_train)
X_test = SimpleImputer().fit_transform(X_test)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Logistic Regression Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# ==== Regression (Random Forest) ====
income_map = {
    'Under $1,000': 1000, '$1,000 to $2,999': 2500, '$3,000 to $4,999': 4000, '$5,000 to $6,999': 6000,
    '$7,000 to $9,999': 8000, '$10,000 to $14,999': 12500, '$15,000 to $19,999': 17500,
    '$20,000 to $24,999': 22500, '$25,000 to $29,999': 27500, '$30,000 to $34,999': 32500,
    '$35,000 to $39,999': 37500, '$40,000 to $49,999': 45000, '$50,000 to $59,999': 55000,
    '$60,000 to $74,999': 67500, '$75,000 to $89,999': 82500, '$90,000 to $119,999': 105000,
    '$120,000 to $159,999': 140000, '$160,000 to $199,999': 180000, '$200,000 to $249,999': 225000,
    '$250,000 or over': 250000
}
df['data_of_earning'] = df['data_of_earning'].map(income_map)
df.dropna(subset=['data_of_earning', 'col23'], inplace=True)
X = df[['col23']].dropna()
y = df['data_of_earning'].dropna()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print("Random Forest MSE:", mean_squared_error(y_test, y_pred))
print("Random Forest R²:", r2_score(y_test, y_pred))