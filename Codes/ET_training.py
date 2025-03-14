import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
import random

# Load the dataset
file_path = "/Users/mrajaian/Downloads/data_backup_47.csv"  # Update with actual file path
df = pd.read_csv(file_path)

# Stripping spaces from column names
df.columns = df.columns.str.strip()

df = df.query('Et != 0')

np.random.seed(42)

numbers = np.arange(1, 1633)

# Select 10% of the elements randomly (without replacement)
sample_size = int(0.1 * len(numbers))
random_sample = np.random.choice(numbers, size=sample_size, replace=False)

df_filtered = df[~df['seed'].isin(random_sample)]
df_filtered_test = df[df['seed'].isin(random_sample)]


features = df_filtered.drop(columns=['Et', 'Ev', 'seed', 'city'])
target = df_filtered['Et']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

et_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
lgb_params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'num_leaves': [20, 31, 50],
    'max_depth': [-1, 10, 20]
}
ann_params = {
    'hidden_layer_sizes': [(128, 64, 32), (256, 128, 64), (64, 32)],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01],
    'max_iter': [200, 500, 1000]
}

# Randomized Grid Search and Model Training
def train_best_model(model, params, name):
    search = RandomizedSearchCV(model, params, n_iter=10, cv=cv, scoring='r2', n_jobs=-1, random_state=42)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    return {
        'Best Params': search.best_params_,
        'R2 Score': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred)
    }

results = {
    'Extra Trees': train_best_model(ExtraTreesRegressor(random_state=42, n_jobs=-1), et_params, 'Extra Trees'),
    'Random Forest': train_best_model(RandomForestRegressor(random_state=42, n_jobs=-1), rf_params, 'Random Forest'),
    'LightGBM': train_best_model(lgb.LGBMRegressor(random_state=42), lgb_params, 'LightGBM'),
    'ANN': train_best_model(MLPRegressor(activation='relu', solver='adam', random_state=42), ann_params, 'ANN')
}

results_df = pd.DataFrame(results).T

print(results_df)
