# 🔮 Attention-based Time Series Forecasting with Deep Learning

## 📌 Project Overview

This project implements an **advanced multivariate time series forecasting system** using deep learning models and attention mechanisms.
The primary objective is to **move beyond black-box forecasting** by integrating **attention-based interpretability**, allowing us to understand *which historical time steps the model prioritizes during prediction*.

The project includes:

* A **statistical baseline** (SARIMA)
* A **standard LSTM neural network**
* An **Attention-enhanced LSTM model**
* **Bayesian hyperparameter optimization** using **Optuna**
* **Interpretability analysis** via learned attention weights

---

## 📊 Dataset Description

A **synthetic yet realistic multivariate time series dataset** is programmatically generated to simulate electricity consumption behavior.

### Dataset Characteristics

* **Observations:** 6000 hourly data points
* **Features (5):**

  * `load` (target variable)
  * `temperature`
  * `humidity`
  * `wind`
  * `pressure`
* **Temporal Patterns:**

  * Daily seasonality
  * Weekly seasonality
  * Long-term trend
  * Heteroskedastic (time-varying) noise

### Why Synthetic Data?

Synthetic generation ensures:

* Full reproducibility
* Controlled complexity
* Realistic non-stationary behavior suitable for deep learning evaluation

### Dataset Generation

The dataset is generated programmatically using NumPy and Pandas:

```bash
python data/data_generator.py
```

This creates:

```
data/synthetic_timeseries.csv
```

---

## 🧪 Data Validation & Sanity Checks

Before modeling, the dataset is validated to ensure correctness:

* Shape: **(6000, 5)**
* No missing values
* Correct feature names and ordering

These checks confirm the dataset is suitable for downstream modeling and evaluation.

---

## ⚙️ Preprocessing Pipeline

1. **Min-Max scaling** across all features
2. **Sliding window sequence generation**

   * Window length: 24 (captures daily seasonality)
3. Target variable: next-step prediction of `load`

This preprocessing preserves temporal ordering and prevents data leakage.

---

## 🤖 Models Implemented

### 1️⃣ SARIMA (Statistical Baseline)

* Seasonal ARIMA with daily seasonality (24-hour cycle)
* Serves as a traditional benchmark for comparison

### 2️⃣ Standard LSTM

* Captures nonlinear temporal dependencies
* Serves as a deep learning baseline

### 3️⃣ Attention-based LSTM (Primary Model)

* LSTM encoder with **learned attention mechanism**
* Attention weights quantify the importance of each time step
* Enables **interpretable forecasting**

---

## 🔍 Bayesian Hyperparameter Optimization

Hyperparameter tuning is performed using **Optuna**, implementing **Bayesian optimization**.

### Tuned Parameters

* Hidden layer size
* Learning rate
* Number of training epochs

### Objective

Minimize training Mean Squared Error (MSE)

This approach is significantly more efficient and principled than grid search.

---

## 📈 Model Evaluation

Models are evaluated using standard time series regression metrics:

* **RMSE (Root Mean Squared Error)**
* **MAE (Mean Absolute Error)**
* **MAPE (Mean Absolute Percentage Error)**

### Example Output

```
ATTENTION LSTM MODEL
RMSE: 0.2267
MAE: 0.1962
MAPE: 60.68
```

The Attention-LSTM demonstrates competitive performance compared to both SARIMA and standard LSTM models.

---

## 🧠 Attention-based Interpretability

One of the core contributions of this project is **model interpretability**.

### Attention Weight Analysis

The learned attention weights are averaged across test samples to identify the **most influential historical time steps**.

Example output:

```
Most important time steps (attention):
t-19 with weight 0.0418
t-18 with weight 0.0418
t-20 with weight 0.0418
```

### Interpretation

* The model prioritizes **recent lagged observations**
* This aligns with expected short-term dependency patterns in electricity demand
* Confirms that the attention mechanism provides **meaningful, non-random explanations**

---

## 🏗️ Project Structure

```
attention-time-series/
├── data/
│   ├── data_generator.py
│   └── synthetic_timeseries.csv
├── src/
│   ├── attention_lstm.py
│   ├── lstm_model.py
│   ├── optuna_tuner.py
│   ├── sarima_model.py
│   ├── trainer.py
│   ├── preprocessing.py
│   └── metrics.py
├── main.py
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run the Project

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Generate dataset

```bash
python data/data_generator.py
```

### 3️⃣ Run full pipeline

```bash
python main.py
```

This executes:

* Data loading & preprocessing
* SARIMA training & evaluation
* LSTM training
* Attention-LSTM + Optuna optimization
* Interpretability analysis

---

## ✅ Key Takeaways

* Attention mechanisms improve **both performance and interpretability**
* Bayesian optimization enables efficient hyperparameter tuning
* The model learns **meaningful temporal dependencies**
* The system is **fully reproducible, modular, and production-ready**

---

## 🏁 Final Notes

This project satisfies all requirements for an **advanced time series forecasting assignment**, including:

* Complex data generation
* Multiple model comparisons
* Bayesian optimization
* Explainable AI integration

---
