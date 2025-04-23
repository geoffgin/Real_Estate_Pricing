import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import pickle

# Load the processed data
df = pd.read_csv('nyc_real_estate_processed.csv')

# Remove outliers
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

# Create features
df['UNITS_PER_SQFT'] = df['TOTAL UNITS'] / df['GROSS SQUARE FEET']
df['COMMERCIAL_RATIO'] = df['COMMERCIAL UNITS'] / df['TOTAL UNITS'].replace(0, 1)
df['LAND_TO_GROSS_RATIO'] = df['LAND SQUARE FEET'] / df['GROSS SQUARE FEET']

# Select features
features = [
    'LOG_GROSS_SQFT',
    'LOG_LAND_SQFT',
    'BUILDING_AGE',
    'UNITS_PER_SQFT',
    'COMMERCIAL_RATIO',
    'LAND_TO_GROSS_RATIO',
    'YEAR BUILT',
    'BOROUGH_NAME',
    'BUILDING_TYPE'
]

# Create dummy variables
df_encoded = pd.get_dummies(df[features], columns=['BOROUGH_NAME', 'BUILDING_TYPE'])

# Prepare X and y
X = df_encoded
y = df['LOG_PRICE_PER_SQFT']

# Scale features
numeric_features = ['LOG_GROSS_SQFT', 'LOG_LAND_SQFT', 'BUILDING_AGE', 
                   'UNITS_PER_SQFT', 'COMMERCIAL_RATIO', 'LAND_TO_GROSS_RATIO', 
                   'YEAR BUILT']
scaler = RobustScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# Train model
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

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Save model and necessary info
with open('nyc_price_model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'features': X.columns.tolist(),
        'scaler': scaler,
        'numeric_features': numeric_features
    }, f)

print("Model trained and saved!")
print("\nFeatures used:", X.columns.tolist())
print("\nNumeric features scaled:", numeric_features)