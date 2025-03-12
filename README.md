# Daylight Prediction Model

This repository contains a trained machine learning model developed to predict daylight illumination values (`Et`) across a sensor grid matrix, leveraging environmental, geographic, and spatial features.

## Project Overview

### Objective
The goal is to accurately predict illumination (`Et`) values for a 25×20 sensor grid to assist architectural and environmental analyses in building design.

### Dataset
- Total dataset size: **484,000** rows.
- Features include weather conditions, solar geometry, spatial sensor information, and geographic attributes.

### Features Used
- **Global Features**:
  - Weather data: `Dir-wea`, `Diff-wea`, `glob-wea`, `Total-Sky-cover`
  - Solar geometry: `Altitude`, `Azimuth`
  - Geographic/environmental data: `dry_bulb_temperature`, `relative_humidity`, `wind_speed`, `latitude`, `longitude`, `elevation`

- **Sensor-Specific Features**:
  - Distances: `SP-South-Dis`, `SP-East-Dis`, `SP-North-Dis`, `SP-West-Dis`, `SP-Ap-Dis`
  - Sensor angle: `Angle`

### Target Variable
- Illumination value (`Et`)

## Machine Learning Model

- **Algorithm**: Extra Trees Regressor
- **Hyperparameters**:
  - `n_estimators`: 50
  - `min_samples_split`: 2
  - `min_samples_leaf`: 1
  - `max_features`: None (All features used)
  - `max_depth`: 15
  - `random_state`: 42
  - `n_jobs`: -1 (using all processors)

## Model Performance

| Metric | Score |
|--------|-------|
| R²     | **Reported in script output** |
| MAE    | **Reported in script output** |
| MSE    | **Reported in script output** |

*(Check the script or notebook for specific numeric performance results.)*

## How to Use the Model

1. **Clone the Repository**
   ```bash
   git clone <repository_link>
   ```

2. **Load the Model**
   ```python
   import joblib

   model = joblib.load('extra_trees_model.pkl')
   ```

3. **Predict**
   ```python
   predictions = model.predict(X_new)
   ```

## Dependencies
- Python 3.x
- pandas
- scikit-learn
- numpy
- joblib

Install dependencies:
```bash
pip install pandas scikit-learn numpy joblib
```

## Author
- [Your Name]

