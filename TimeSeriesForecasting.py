####Train random forest model
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from math import sqrt
import time

# Load the dataset
df1 = pd.read_csv('../input/sales-forecasting-womart-store/TRAIN.csv')
df= df1.drop_duplicates()

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Extract time-related features
df['Week'] = df['Date'].dt.isocalendar().week
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Drop the "ID" and "Date" columns
df = df.drop(['ID', 'Date'], axis=1)

# Split the data into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Define categorical columns to be one-hot encoded along with new time-related features
categorical_cols = ['Store_Type', 'Location_Type', 'Region_Code', 'Discount', 'Week', 'Day', 'Month', 'Year', 'Holiday']

# Create a ColumnTransformer for one-hot encoding
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(drop='first', sparse=False), categorical_cols)],
    remainder='passthrough'
)

# Fit the ColumnTransformer on the training data and transform both training and test data
X_train = preprocessor.fit_transform(train_data.drop('Sales', axis=1))
X_test = preprocessor.transform(test_data.drop('Sales', axis=1))
y_train = train_data['Sales']
y_test = test_data['Sales']

# Train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=50, max_depth=5)
rf_model.fit(X_train, y_train)
predictions_rf = rf_model.predict(X_test)

# Evaluate the  model
rf_rmse = sqrt(mean_squared_error(y_test, predictions_rf))
print(f"Random Forest RMSE: {rf_rmse}")




###tune hyperparameters with grid search for random forest
from sklearn.model_selection import GridSearchCV

# Step 1: Tune 'n_estimators' keeping 'max_depth' at a reasonable fixed value (e.g., 5)

# Create the parameter grid for 'n_estimators'
param_grid_1 = {
    'n_estimators': [50, 100, 200]
}

# Create a based model
rf = RandomForestRegressor(max_depth=5)

# Instantiate the grid search model
grid_search_1 = GridSearchCV(estimator = rf, param_grid = param_grid_1, 
                          cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search_1.fit(X_train, y_train)

# Get the best 'n_estimators' parameter
best_n_estimators = grid_search_1.best_params_['n_estimators']

best_n_estimators

# Step 2: Tune 'max_depth' using the best 'n_estimators' value found (which is 100)

# Create the parameter grid for 'max_depth'
param_grid_2 = {
    'max_depth': [5, 10, 15]
}

# Create a based model with the best 'n_estimators'
rf = RandomForestRegressor(n_estimators=100)  # Using 100 as the best_n_estimators value

# Instantiate the grid search model
grid_search_2 = GridSearchCV(estimator = rf, param_grid = param_grid_2, 
                          cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search_2.fit(X_train, y_train)

# Get the best 'max_depth' parameter
best_max_depth = grid_search_2.best_params_['max_depth']

# Train the Random Forest model with best parameters
best_rf_model = RandomForestRegressor(n_estimators=100, max_depth=best_max_depth)
best_rf_model.fit(X_train, y_train)
predictions_best_rf = best_rf_model.predict(X_test)

# Evaluate the tuned model
best_rf_rmse = sqrt(mean_squared_error(y_test, predictions_best_rf))
print(f"Tuned Random Forest RMSE: {best_rf_rmse}")




### Train XGBoost model

# Train the XGBoost model
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=50, max_depth=5)
xgb_model.fit(X_train, y_train)
predictions_xgb = xgb_model.predict(X_test)

# Evaluate the  model
xgboost_rmse = sqrt(mean_squared_error(y_test, predictions_xgb))
print(f"XGboost RMSE: {xgboost_rmse}")

#Tunning XGBoost
# Step 1: Tune 'n_estimators' keeping 'max_depth' and 'learning_rate' at default values

# Create the parameter grid for 'n_estimators'
param_grid_xgb_1 = {
    'n_estimators': [50, 100, 200]
}

# Create a based model
xgb = XGBRegressor(objective='reg:squarederror')

# Instantiate the grid search model
grid_search_xgb_1 = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb_1,
                                 cv=3, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search_xgb_1.fit(X_train, y_train)

# Get the best 'n_estimators' parameter
best_n_estimators_xgb = grid_search_xgb_1.best_params_['n_estimators']
best_n_estimators_xgb

# Step 2: Tune 'max_depth' using the best 'n_estimators' value found (which is 200)

# Create the parameter grid for 'max_depth'
param_grid_xgb_2 = {
    'max_depth': [3, 5, 7]
}

# Create a based model with the best 'n_estimators'
xgb = XGBRegressor(objective='reg:squarederror', n_estimators=200)  # Using 200 as the best_n_estimators_xgb value

# Instantiate the grid search model
grid_search_xgb_2 = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb_2,
                                 cv=3, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search_xgb_2.fit(X_train, y_train)

# Get the best 'max_depth' parameter
best_max_depth_xgb = grid_search_xgb_2.best_params_['max_depth']

# Train the XGBoost model with best parameters
best_xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=200, max_depth=best_max_depth_xgb)
best_xgb_model.fit(X_train, y_train)
predictions_best_xgb = best_xgb_model.predict(X_test)

# Evaluate the tuned model
best_xgb_rmse = sqrt(mean_squared_error(y_test, predictions_best_xgb))
best_xgb_rmse
print(f"Tuned XGBoost RMSE: {best_rf_rmse}")

#Tune regularization terms (reg_alpha and reg_lambda using the best n_estimators and max_depth)
# Step 3: Tune 'reg_alpha' and 'reg_lambda' using the best 'n_estimators' and 'max_depth' values found

# Create the parameter grid for 'reg_alpha' and 'reg_lambda'
param_grid_xgb_3 = {
    'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05],
    'reg_lambda': [0.5, 1, 1.5, 2]
}

# Create a based model with the best 'n_estimators' and 'max_depth'
xgb = XGBRegressor(objective='reg:squarederror', n_estimators=best_n_estimators_xgb, max_depth=best_max_depth_xgb)

# Instantiate the grid search model
grid_search_xgb_3 = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb_3,
                                 cv=3, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search_xgb_3.fit(X_train, y_train)

# Get the best 'reg_alpha' and 'reg_lambda' parameters
best_reg_alpha_xgb = grid_search_xgb_3.best_params_['reg_alpha']
best_reg_lambda_xgb = grid_search_xgb_3.best_params_['reg_lambda']

# Train the XGBoost model with best parameters
best_xgb_model = XGBRegressor(objective='reg:squarederror', 
                              n_estimators=best_n_estimators_xgb, 
                              max_depth=best_max_depth_xgb,
                              reg_alpha=best_reg_alpha_xgb,
                              reg_lambda=best_reg_lambda_xgb)

best_xgb_model.fit(X_train, y_train)
predictions_best_xgb = best_xgb_model.predict(X_test)

# Evaluate the tuned model
best_xgb_rmse = sqrt(mean_squared_error(y_test, predictions_best_xgb))
print(f"Tuned XGBoost RMSE: {best_xgb_rmse}")




###Ensemble models before tunning
# Ensemble the predictions
final_predictions = (predictions_rf + predictions_xgb) / 2

# Evaluate the ensemble model
ensemble_rmse = sqrt(mean_squared_error(y_test, final_predictions))
print(f"Ensemble RMSE: {ensemble_rmse}")


###Ensemble models after tunning with weighted average
# Calculate the inverse of each model's RMSE as the weight
weight_xgb = 1 / best_xgb_rmse
weight_rf = 1 / best_rf_rmse
# Calculate the total weight
total_weight = weight_xgb + weight_rf

# Normalize the weights
normalized_weight_xgb = weight_xgb / total_weight
normalized_weight_rf = weight_rf / total_weight

# Ensemble the predictions
ensemble_tuned_predictions = (predictions_best_rf*normalized_weight_rf) + (predictions_best_xgb*normalized_weight_xgb)

# Evaluate the ensemble model
ensemble_tuned_rmse = sqrt(mean_squared_error(y_test, ensemble_tuned_predictions))

ensemble_tuned_rmse

