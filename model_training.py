import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import pickle

# Load the processed data
df = pd.read_csv('nyc_real_estate_processed.csv')

# Remove extreme outliers
Q1 = df['PRICE_PER_SQFT'].quantile(0.01)
Q3 = df['PRICE_PER_SQFT'].quantile(0.99)
IQR = Q3 - Q1
df = df[(df['PRICE_PER_SQFT'] >= Q1 - 1.5 * IQR) & 
        (df['PRICE_PER_SQFT'] <= Q3 + 1.5 * IQR)]

# Log transform the target variable
df['LOG_PRICE_PER_SQFT'] = np.log1p(df['PRICE_PER_SQFT'])

# Handle missing values
df['LAND SQUARE FEET'].fillna(df['LAND SQUARE FEET'].median(), inplace=True)
df['LOG_LAND_SQFT'].fillna(df['LOG_LAND_SQFT'].median(), inplace=True)
df['AGE_CAT'].fillna(df['AGE_CAT'].mode()[0], inplace=True)

# Create new features
df['UNITS_PER_SQFT'] = df['TOTAL UNITS'] / df['GROSS SQUARE FEET']
df['COMMERCIAL_RATIO'] = df['COMMERCIAL UNITS'] / df['TOTAL UNITS'].replace(0, 1)
df['LAND_TO_GROSS_RATIO'] = df['LAND SQUARE FEET'] / df['GROSS SQUARE FEET']

# Select features
features = [
    'LOG_GROSS_SQFT',
    'LOG_LAND_SQFT',
    'BUILDING_AGE',
    'BOROUGH_NAME',
    'BUILDING_TYPE',
    'IS_MIXED_USE',
    'UNITS_PER_SQFT',
    'COMMERCIAL_RATIO',
    'LAND_TO_GROSS_RATIO',
    'YEAR BUILT',
    'AGE_CAT',
    'SIZE_CAT',
    'NEIGHBORHOOD'  # Adding neighborhood for more local context
]

# Create dummy variables
df_encoded = pd.get_dummies(df[features])

# Prepare X and y
X = df_encoded
y = df['LOG_PRICE_PER_SQFT']

# Scale features
scaler = RobustScaler()
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42
)

# Perform cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print("\nCross-validation scores:", cv_scores)
print("Average CV score:", cv_scores.mean())

# Train final model
model.fit(X_train, y_train)

# Print model performance
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"\nTrain R² Score: {train_score:.4f}")
print(f"Test R² Score: {test_score:.4f}")

# Save the model and necessary transformations
with open('nyc_price_model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'features': X.columns.tolist(),
        'scaler': scaler,
        'feature_list': features
    }, f)

# Print feature importances
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Print example predictions with error percentages
print("\nExample Predictions:")
y_pred = model.predict(X_test[:5])
y_test_actual = np.expm1(y_test[:5])
y_pred_actual = np.expm1(y_pred)

print("Actual vs Predicted prices per sq ft:")
for actual, pred in zip(y_test_actual, y_pred_actual):
    error_pct = abs(actual - pred) / actual * 100
    print(f"Actual: ${actual:.2f}, Predicted: ${pred:.2f}, Error: {error_pct:.1f}%")