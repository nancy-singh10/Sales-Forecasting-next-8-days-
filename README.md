# End-to-End Time Series Forecasting System

## Problem Statement
The objective of this project is to build a production-ready forecasting system to predict the next 8 weeks of sales for each state based on historical sales data. 

## Dataset
The dataset contains historical sales with the following columns:
- `State`: The US state where sales occurred.
- `Date`: Weekly snapshot of the data.
- `Total`: The total sales volume (in string format with commas).
- `Category`: The item category (e.g. Beverages).

## Architecture
The final production architecture is as follows:
CSV Dataset -> Preprocessing Pipeline -> Feature Engineering -> Model Training -> Evaluation Engine -> Best Model Selection -> Model Registry -> Prediction Service -> FastAPI -> Docker -> Deployment

## Models Evaluated
1. **SARIMA**: Traditional statistical model for capturing seasonality and trend.
2. **Prophet**: Additive regression model by Meta for handling holidays and strong seasonal effects.
3. **XGBoost**: Tree-based gradient boosting algorithm utilizing lag and rolling features.
4. **LSTM**: Deep learning Recurrent Neural Network for capturing complex temporal dependencies.

## Metrics
The models are evaluated using **RMSE** (Root Mean Squared Error) and **MAPE** (Mean Absolute Percentage Error).

## API Endpoints
The REST API is built with FastAPI.
- `GET /`: Health check endpoint.
- `GET /forecast/{state}`: Returns the next 8 weeks forecast for the provided state.

## Installation & Setup
1. Clone the repository and initialize the environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Train the models and evaluate:
   ```bash
   python src/train_sarima.py
   python src/train_prophet.py
   python src/train_xgboost.py
   python src/train_lstm.py
   python src/evaluate_models.py
   ```
3. Run the FastAPI Application:
   ```bash
   uvicorn api.app:app --reload
   ```

## Docker Usage
To build and run the docker container:
```bash
docker build -t forecasting-system .
docker run -p 8000:8000 forecasting-system
```

## Future Improvements
- Implement automated hyperparameter tuning (e.g., Optuna) for XGBoost and LSTM.
- Use a dedicated database (PostgreSQL) instead of flat CSV files.
- Add Airflow or Prefect for pipeline orchestration and automated retraining.
