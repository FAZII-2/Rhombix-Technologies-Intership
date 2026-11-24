# House Price Prediction using Machine Learning

## Overview
This project builds a regression model that predicts the median house price from features such as location, house age, number of rooms, population and income. It covers the full pipeline: data loading, exploratory data analysis (EDA), cleaning, feature engineering, model comparison, hyperparameter tuning, evaluation, and a small interactive dashboard for predictions.

## Dataset
- **Name:** California Housing Prices (Kaggle)
- **File:** `housing.csv`
- **Size:** 20,640 rows originally, 19,548 rows after cleaning
- **Target:** `median_house_value`
- **Features:** longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity

## Project Workflow
1. **Data inspection:** Checked shape, data types, missing values, duplicates and basic statistics.
2. **Exploratory Data Analysis:**
   - Price distribution, correlation heatmap and income vs price scatter plot
   - Price map using latitude and longitude
   - Price comparison across location categories, income groups and house age groups
   - Outlier and skewness checks
3. **Data cleaning:**
   - Removed 992 rows where the price was capped at 500,001 (a data limit, not a real price)
   - Filled missing `total_bedrooms` values with the median
   - Removed 100 extreme outlier rows
4. **Feature engineering:**
   - `rooms_per_household`
   - `bedrooms_per_room`
   - `population_per_household`
   - One-hot encoding of `ocean_proximity`
5. **Train/test split:** 80% training (15,638 rows) and 20% testing (3,910 rows), 16 features.
6. **Model training:** Linear Regression, Decision Tree and XGBoost were trained and compared.
7. **Hyperparameter tuning:** `RandomizedSearchCV` (15 combinations, 3-fold cross-validation) on XGBoost.
8. **Evaluation:** R2, MAE, RMSE, percentage error, feature importance and error plots.
9. **Dashboard:** Sliders and a Predict button built with `ipywidgets`.

## Results

### Model Comparison (before tuning)
| Model | R2 | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.648 | $42,935 | $58,605 |
| Decision Tree | 0.678 | $37,854 | $56,036 |
| **XGBoost** | **0.824** | **$27,904** | **$41,463** |

XGBoost performed best and was selected for tuning.

### Hyperparameter Tuning (XGBoost)
**Best parameters:**
| Parameter | Value |
|---|---|
| n_estimators | 400 |
| max_depth | 8 |
| learning_rate | 0.05 |
| subsample | 0.85 |
| colsample_bytree | 0.7 |
| min_child_weight | 3 |

| Metric | Before Tuning | After Tuning |
|---|---|---|
| R2 | 0.8236 | **0.8371** |
| MAE | $27,904 | **$26,297** |
| RMSE | $41,463 | **$39,845** |

### Final Model Performance
- **R2:** 0.8371
- **MAE:** $26,297
- **RMSE:** $39,845
- **Median percentage error:** 10.6%
- **Predictions within 10% of the actual price:** 47.9%
- **Predictions within 20% of the actual price:** 76.1%

### Top Feature Importances
| Feature | Importance |
|---|---|
| ocean_proximity_INLAND | 0.6053 |
| median_income | 0.0765 |
| ocean_proximity_NEAR_BAY | 0.0585 |
| ocean_proximity_ISLAND | 0.0391 |
| ocean_proximity_NEAR_OCEAN | 0.0354 |
| population_per_household | 0.0302 |
| longitude | 0.0279 |
| ocean_proximity_1H_OCEAN | 0.0278 |

## Key Insights
- **Location is the biggest price driver.** Inland houses average about $125k, while coastal areas average $240k to $260k, so the inland flag is the most important feature.
- **Median income is the strongest numeric feature** (correlation of 0.688 with price).
- **XGBoost beats Linear Regression by a wide margin** because the relationship between location and price is non-linear.
- **Tuning improved the model**, raising R2 from 0.8236 to 0.8371 and reducing RMSE by about $1,600.

## Dashboard
The last cell of the notebook provides sliders for income, house age, rooms, bedrooms, population, households, latitude and longitude, plus a location dropdown. Clicking **Predict Price** shows the estimated house price.

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn
- ipywidgets
- Kaggle Notebook

## How to Run
1. Open the notebook in Kaggle.
2. Add the **California Housing Prices** dataset as an input.
3. Run all cells in order.
4. Use the dashboard in the last cell to test your own inputs.

## Limitations and Future Work
- The dataset contains district-level medians from the California census, so predictions are estimates for an area, not for an individual house.
- The ISLAND category has only 5 rows, so its importance is not reliable.
- About 24% of predictions are off by more than 20%, so there is room for improvement.
- Future work could include a wider hyperparameter search, additional features such as distance to major cities, and a web deployment using Streamlit.

## Author
Faizan Ali
Internship Project: Rhombix Technologies, 
