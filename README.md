# AQI Predictor: Karachi Air Quality Forecasting Service

This project is a **100% serverless** end-to-end AQI prediction service that provides 3-day air quality forecasts for **Karachi, Pakistan**. It leverages automated data pipelines and machine learning to maintain a continuous cycle of hourly data ingestion and daily model retraining.

## Architecture Diagram
The following diagram illustrates the data flow from the Open Meteo API through feature engineering, automated training, and final deployment.

![Architecture Diagram](architecture_overview.png)

---

## Core Technology Stack
* **Data Collection**: Open Meteo API.
* **Feature Engineering**: Numpy and Pandas.
* **Exploratory Data Analysis**: Matplotlib and Seaborn.
* **Storage & Registry**: Hopsworks Feature Store and Model Registry.
* **Machine Learning**: Scikit-Learn and SHAP for feature importance analysis.
* **Deployment**: Streamlit and GitHub Actions.

---

## Methodology

### A. Backfill Data (Initial Setup)
* Fetches the last 8 months of raw pollutant data from the Open Meteo API.
* Generates time-based features: `day`, `month`, `year`, `day of week`, and `weekend`.
* Stores initial features in the Hopsworks feature group.
* Executed only once.

### B. Hourly Data Ingestion
* **Automation**: Runs every hour via GitHub Actions.
* Fetches latest hourly raw pollutant data from Open Meteo API.
* Creates features according to the established schema and stores them in Hopsworks.

### C. Daily Training Pipeline
* **Automation**: Runs every day via GitHub Actions.
* Fetches historical features and targets from the Hopsworks feature group.
* **Models Trained**: Random Forest Regressor, Gradient Boosting Regressor, and Ridge Regressor.
* **Evaluation**: Models are evaluated using $R^2$, MAE, MAPE, and RMSE metrics.
* The best models and their metrics are saved in the model registry.

### D. Deployment
* A user-friendly dashboard displays real-time and forecasted pollutant levels.
* The application is hosted on Streamlit Cloud using a GitHub repository.
* **Live App**: [air-quality-predictor-project.streamlit.app](https://air-quality-predictor-project.streamlit.app/) 

---

## Dashboard Preview

![Dashboard1](/dashboard_screenshot_1.png)
![Dashboard2](/dashboard_screenshot_2.png)
---


## Future Improvements
* Optimize features and model tuning to reduce overfitting.
* Implement automated hazardous AQI alerts for user safety.
* Integrate additional data sources to improve predictive power.
