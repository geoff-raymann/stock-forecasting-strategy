# 📈 Stock Price Forecasting & Investment Strategy

This project builds a robust stock price forecasting and risk analysis tool using ARIMA, LSTM, and Prophet models. It supports investment decision-making through Monte Carlo simulations based on Geometric Brownian Motion (GBM) and offers an interactive dashboard for analysis.

---

## 🧠 Objectives

- Forecast the next **30–60 trading days** of a selected stock (e.g., AAPL).
- Evaluate and compare **ARIMA**, **LSTM**, and **Prophet** forecasting models.
- Run **Monte Carlo simulations** to model price evolution and investment risk.
- Deploy a dashboard for **interactive forecasting and simulation**.

---

## 🛠️ Technologies Used

- **Language**: Python 3.10+
- **Libraries**: 
  - Data Handling: `pandas`, `numpy`
  - Modeling: `statsmodels`, `tensorflow`, `keras`, `prophet`
  - Visualization: `matplotlib`, `seaborn`, `plotly`
  - Evaluation: `sklearn.metrics`
- **Deployment**: Streamlit or Dash (TBD)
- **Notebook**: Jupyter for exploratory analysis

---

## 🗂️ Project Structure

```plaintext
stock-forecasting-strategy/
├── data/               # Input CSV files (e.g. AAPL.csv)
├── src/
│   ├── arima_model.py
│   ├── lstm_model.py
│   ├── prophet_model.py
│   ├── evaluate_models.py
│   └── monte_carlo_simulation.py
├── results/            # Forecasts, metrics, and generated plots
├── notebooks/          # Jupyter notebooks for prototyping
├── app/                # Streamlit or Dash app code
└── README.md
🚀 Getting Started
1. Clone the Repository

git clone https://github.com/your-username/stock-forecasting-strategy.git
cd stock-forecasting-strategy
2. Create Virtual Environment

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies

pip install -r requirements.txt
ℹ️ Note: Ensure Prophet installs correctly. You may need pystan and cmdstanpy.

📊 How to Use
Train & Forecast
Run any of the model scripts individually:


python src/arima_model.py
python src/lstm_model.py
python src/prophet_model.py
Evaluate Model Accuracy

python src/evaluate_models.py
Run Monte Carlo Simulation

python src/monte_carlo_simulation.py
📈 Forecasting Models
Model	Description	Strengths
ARIMA	Time-series with trend/seasonality	Strong for linear and stationary data
LSTM	Recurrent neural network	Captures long-term dependencies, nonlinear
Prophet	Additive model from Meta/Facebook	Handles holidays, missing data, trends

🎲 Monte Carlo Simulation
Uses Geometric Brownian Motion (GBM) to simulate thousands of possible future price paths. Outputs include:

Price distribution histograms

VaR (Value at Risk)

Expected returns & volatility

📊 Dashboard Features (Coming Soon)
Ticker symbol input (e.g., AAPL)

Forecast chart with model comparison

Monte Carlo simulation interface

Investment risk-adjustment slider

Export forecasts & visualizations

✅ Evaluation Metrics
Metric	Description
MAE	Mean Absolute Error
MSE	Mean Squared Error
RMSE	Root Mean Squared Error

All models are benchmarked using a unified evaluate_models.py script.

📌 TODOs
 Model forecasts (ARIMA, LSTM, Prophet)

 Centralized evaluation script

 Monte Carlo simulation

 Streamlit/Dash dashboard

 Ticker selection and forecasting range UI

 Dockerize for deployment

📚 References
Yahoo Finance API

Geometric Brownian Motion - Investopedia

Facebook Prophet Docs

👨‍💻 Author
Geoffrey Odiwuor
ALX Data Science Graduate | Portfolio Project – 2025
LinkedIn | GitHub

📄 License
MIT License – See LICENSE for details.

