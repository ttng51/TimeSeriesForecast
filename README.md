# Time Series Forecast with XGBoost and Random Forest
### Overview
This repository contains Python code and datasets used in the blog post "Fine-Tuning Machine Learning Models for Time Series Forecasting in Sales". The project focuses on predicting sales using machine learning algorithms like Random Forest and XGBoost. The primary objective is to fine-tune the models to achieve the best forecasting based on Root Mean Square Error (RMSE).

### Installation and Requirements
- Python 3.x
- pandas
- scikit-learn
- XGBoost
### Data Preprocessing
The data preprocessing steps include:
- Handling duplicate values
- Converting date fields into usable formats (day, month, year)
- One-hot encoding for categorical variables
### Model Building
Two machine learning algorithms were utilized:
#### Random Forest
- Ensemble of Decision Trees
- Implemented using scikit-learn's RandomForestRegressor
#### XGBoost
- Gradient Boosting model
- Implemented using the XGBoost Python library
### Fine-Tuning
Both models were fine-tuned to optimize their performance. The hyperparameters like n_estimators, max_depth, and regularization terms (reg_alpha, reg_lambda) for XGBoost were tuned using GridSearchCV.
### Results
The models were evaluated based on RMSE, with the XGBoost model achieving the best performance with an RMSE of ~2014.
### License
This project is licensed under the MIT License - see the LICENSE.md file for details.
