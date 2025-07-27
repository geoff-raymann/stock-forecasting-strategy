# To be completed: walk-forward RMSE/MAE over N windows
# Later integrate into evaluate_models.py

def rolling_backtest(ticker, model_func, window=30, step=5):
    # Load data, split into rolling windows, run model_func, store metrics
    # model_func = callable that returns forecast for last window
    pass
