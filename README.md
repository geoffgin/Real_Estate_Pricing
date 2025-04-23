# NYC Real Estate Price Predictor

A machine learning model to predict commercial real estate prices in New York City. This project includes both the model training code and a user-friendly web interface for making predictions.

![Streamlit Interface](assets/streamlit_interface.png)

## Features

- Predicts price per square foot and total price for properties
- Covers all five NYC boroughs
- Accounts for multiple factors:
  - Property size (gross square feet)
  - Land size
  - Building age
  - Building type (Residential/Commercial/Other)
  - Location (Borough)
- User-friendly web interface built with Streamlit
- Model trained on NYC Department of Finance rolling sales data

## Model Performance

- Test R² Score: 0.3770
- Average prediction error: ~15%
- Key predictive features:
  - Property size
  - Location (Borough)
  - Building age
  - Land-to-building ratio

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nyc-real-estate-predictor.git
cd nyc-real-estate-predictor

