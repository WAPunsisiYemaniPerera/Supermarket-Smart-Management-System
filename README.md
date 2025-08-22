# Supermarket Smart Management System 🛒

A comprehensive data analysis and machine learning project that leverages sales data to build predictive models for a supermarket. This repository contains the code and analysis for forecasting profit, identifying potential loyalty members, and predicting item demand.

## 🎯 Project Goals

This project aims to provide actionable insights for supermarket management through three key predictive models:

1.  **📈 Profit Forecasting:** A regression model to predict the total profit for the upcoming month, enabling better financial planning and performance assessment.
2.  **👑 Loyalty Member Identification:** A classification model to identify customers who exhibit loyalty-member behavior but are not yet part of the program, allowing for targeted marketing campaigns.
3.  **📦 Demand Forecasting:** A time-series analysis to identify the most popular and trending items, helping to optimize stock and inventory management.

## 🛠️ Technologies & Libraries Used

* **Language:** Python
* **Core Libraries:**
    * `pandas` for data manipulation and analysis.
    * `numpy` for numerical operations.
    * `scikit-learn` for building and evaluating machine learning models.
    * `xgboost` for advanced gradient boosting models.
    * `scipy` for statistical analysis, including trend calculation.
* **Visualization:**
    * `matplotlib` and `seaborn` for data visualization and creating insightful plots.
* **Model Persistence:**
    * `pickle` and `joblib` for saving and loading trained models.

## 📂 Repository Structure

.
├── model_training.ipynb      # Jupyter notebook with all the analysis and model development
├── requirements.txt          # A list of the Python packages required to run this project
├── best_profit_model.pkl     # Saved model for profit prediction
├── best_loyalty_model.pkl    # Saved model for loyalty member identification
└── README.md                 # This file


## 🚀 How to Get Started

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    ```
2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Explore the analysis:**
    Open and run the `model_training.ipynb` notebook in a Jupyter environment to see the full data exploration, preprocessing, model training, and evaluation process.

## ✨ Models in Detail

### 1. Model 1 - Next-Month's Profit Prediction

* **Problem Type:** Regression
* **Goal:** Predict a continuous value (profit).
* **Algorithms Used:** Random Forest Regressor, Gradient Boosting Regressor.
* **Key Features:** This model uses features like historical sales data, promotional discounts, and time-based features (month, day of the week) to make its predictions.

### 2. Model 2 - Potential Loyalty Member Identification

* **Problem Type:** Classification
* **Goal:** Predict a binary category ('Yes' or 'No' for NexusMember).
* **Algorithms Used:** Logistic Regression, Random Forest Classifier, XGBClassifier.
* **Key Features:** The model analyzes customer purchasing behavior, including total spending, frequency of visits, and preferred departments, to identify potential loyalty members.

### 3. Model 3 - Most Demanded Items Prediction

* **Problem Type:** Time-Series Analysis & Trend Identification
* **Goal:** Rank items based on current popularity and future demand trends.
* **Methodology:** This analysis aggregates sales data weekly for each item and calculates a "Trend Score" using linear regression to determine if demand is increasing or decreasing over time.
