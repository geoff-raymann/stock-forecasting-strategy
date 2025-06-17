import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_forecast(y_true, y_pred, model_name="Model"):
    """
    Evaluate a forecast using common error metrics.
    
    Parameters:
        y_true (array-like): True values
        y_pred (array-like): Predicted values
        model_name (str): Name of the model for reporting

    Returns:
        dict: Dictionary of MAE, MSE, RMSE
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    result = {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    }

    return result


def print_evaluation(result_dict):
    """
    Nicely print the evaluation results.

    Parameters:
        result_dict (dict): Output from evaluate_forecast()
    """
    print(f"\n📊 Evaluation for {result_dict['Model']}")
    print("-" * 30)
    print(f"MAE : {result_dict['MAE']:.4f}")
    print(f"MSE : {result_dict['MSE']:.4f}")
    print(f"RMSE: {result_dict['RMSE']:.4f}")


# If you run this script directly, show a demo
if __name__ == "__main__":
    # Sample usage/demo
    y_true = [100, 102, 105, 107, 110]
    y_pred = [101, 103, 104, 106, 111]
    results = evaluate_forecast(y_true, y_pred, model_name="DemoModel")
    print_evaluation(results)
