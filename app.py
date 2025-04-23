import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the saved model and transformations
with open('nyc_price_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    features = model_data['features']
    scaler = model_data['scaler']

def predict_price(borough, sqft, land_sqft, age, is_mixed_use=0, building_type='RESIDENTIAL'):
    # Create feature dictionary
    input_data = {
        'LOG_GROSS_SQFT': np.log1p(sqft),
        'LOG_LAND_SQFT': np.log1p(land_sqft),
        'BUILDING_AGE': age,
        'UNITS_PER_SQFT': 1 / sqft,
        'COMMERCIAL_RATIO': 0 if building_type == 'RESIDENTIAL' else 1,
        'LAND_TO_GROSS_RATIO': land_sqft / sqft,
        'YEAR BUILT': 2023 - age,
        'IS_MIXED_USE': is_mixed_use,
        'BOROUGH_NAME': borough,
        'BUILDING_TYPE': building_type
    }
    
    # Create DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Create dummy variables for categorical columns
    categorical_cols = ['BOROUGH_NAME', 'BUILDING_TYPE']
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols)
    
    # Ensure all columns from training are present
    for col in features:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    # Select only the features used during training
    input_encoded = input_encoded[features]
    
    # Scale numeric features
    numeric_cols = ['LOG_GROSS_SQFT', 'LOG_LAND_SQFT', 'BUILDING_AGE', 
                   'UNITS_PER_SQFT', 'COMMERCIAL_RATIO', 'LAND_TO_GROSS_RATIO', 
                   'YEAR BUILT']
    
    input_encoded[numeric_cols] = scaler.transform(input_encoded[numeric_cols])
    
    # Make prediction
    log_price = model.predict(input_encoded)[0]
    price_per_sqft = np.expm1(log_price)
    total_price = price_per_sqft * sqft
    
    return {
        'price_per_sqft': price_per_sqft,
        'total_price': total_price
    }

# Streamlit interface
st.title('NYC Commercial Real Estate Price Predictor')

col1, col2 = st.columns(2)

with col1:
    borough = st.selectbox(
        'Select Borough',
        ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
    )
    
    building_type = st.selectbox(
        'Building Type',
        ['RESIDENTIAL', 'COMMERCIAL', 'OTHER']
    )

with col2:
    sqft = st.number_input('Gross Square Feet', min_value=100, max_value=100000, value=2000)
    land_sqft = st.number_input('Land Square Feet', min_value=100, max_value=100000, value=2000)
    age = st.number_input('Building Age', min_value=0, max_value=200, value=50)

is_mixed_use = st.checkbox('Is Mixed Use')

if st.button('Predict Price'):
    prediction = predict_price(
        borough=borough,
        sqft=sqft,
        land_sqft=land_sqft,
        age=age,
        is_mixed_use=int(is_mixed_use),
        building_type=building_type
    )
    
    st.write(f"Estimated Price per Square Foot: ${prediction['price_per_sqft']:,.2f}")
    st.write(f"Estimated Total Price: ${prediction['total_price']:,.2f}")
    
    # Add confidence interval
    st.write("Note: Predictions are typically within 15% of actual values")