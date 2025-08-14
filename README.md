# 📈 Stock Forecasting & Strategy Dashboard

A fully-featured forecasting dashboard built with **Streamlit**, showcasing time series predictions using **ARIMA**, **LSTM**, **XGBoost**, **Prophet**, and a dynamic **Ensemble model** powered by inverse RMSE weighting. Designed for executive insights and investment strategy.

---

## 🚀 Features

✅ ARIMA, LSTM, XGBoost, Prophet forecasting models  
✅ Weighted ensemble model (based on inverse RMSE)  
✅ Model evaluation: MAE, MSE, RMSE  
✅ Interactive Streamlit executive dashboard (Plotly charts: hover, zoom, pan)  
✅ Export metrics to Excel and HTML  
✅ Monte Carlo simulation for strategy testing  
✅ Executive KPIs: Best model, RMSE, latest forecast, latest actual, MC stats  
✅ Ready for deployment on Streamlit Cloud  

---

## 📊 Demo

You can try the deployed app here:  
👉 [Streamlit Cloud App URL](https://your-streamlit-cloud-url)

---

## 📁 Project Structure

```
stock-forecasting-strategy/
│
├── data/                        # Raw CSVs (per ticker)
├── results/                     # Model outputs: forecasts and evaluation
│   └── [model]/[ticker]/...
│
├── src/                         # All model scripts (CLI compatible)
│   ├── arima_model.py
│   ├── lstm_model.py
│   ├── xgboost_model.py
│   ├── facebook_prophet_model.py
│   ├── ensemble_model_reverse_rmse.py
│   ├── monte_carlo_simulation.py
│   └── evaluate_models.py
│
├── app/
│   └── dashboard.py             # Streamlit dashboard
│
├── requirements.txt
├── packages.txt
├── README.md
└── .streamlit/config.toml       # Optional: Streamlit theme configs
```

---

## 💻 Local Setup (Conda Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/stock-forecasting-strategy.git
cd stock-forecasting-strategy

# 2. Create and activate conda environment
conda create -n stock-forecasting-env python=3.10
conda activate stock-forecasting-env

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app/dashboard.py
```

---

## ☁️ Deploying to Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and create a new app from your GitHub repo.
3. Ensure the following files are in your repo:
   - `requirements.txt`
   - `packages.txt` (see below)
   - `app/dashboard.py`

### 📦 `packages.txt`

```txt
libgomp1
build-essential
python3-dev
gcc
g++
libatlas-base-dev
```

---

## 📈 Running Forecast Scripts (CLI)

```bash
# Run ARIMA model
python src/arima_model.py --ticker AAPL

# Run LSTM model
python src/lstm_model.py --ticker AAPL

# Run XGBoost model
python src/xgboost_model.py --ticker AAPL

# Run Prophet model
python src/facebook_prophet_model.py --ticker AAPL

# Run Ensemble model (LSTM + XGBoost)
python src/ensemble_model_reverse_rmse.py --ticker AAPL

# Run Monte Carlo Simulation
python src/monte_carlo_simulation.py --ticker AAPL
```

---

## 📤 Export Features

From the dashboard, you can:
- 📥 Export metrics as Excel or CSV
- 📄 Download HTML summary
- 📊 View interactive statistical tables and charts

---

## 🤝 Contributing

Pull requests are welcome. Please fork the repository and create a new branch for major changes.

---

## 📃 License

MIT License. Feel free to use, modify, and share.

---

## 🙌 Acknowledgements

- Streamlit for rapid app deployment  
- Facebook Prophet, TensorFlow, and XGBoost teams  
- ALX Data Science Career Path support  

---

## 👨‍💻 Author

Geoffrey Odiwuor