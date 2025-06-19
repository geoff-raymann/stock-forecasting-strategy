import os
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


def parse_evaluation_file(filepath):
    """
    Parses a saved evaluation.txt file and returns a dictionary
    with model name and performance metrics (MAE, MSE, RMSE).
    """
    metrics = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("MODEL"):
                metrics["Model"] = line.split(":")[1].strip()
            elif any(metric in line.upper() for metric in ["MAE", "MSE", "RMSE"]):
                key, value = line.split(":")
                metrics[key.strip()] = float(value.strip())
    return metrics


def load_all_model_evaluations(results_dir):
    """
    Recursively scans through the results directory and loads all
    *_evaluation.txt files into a single dataframe.

    Parameters:
        results_dir (str): Base directory where model results are stored.

    Returns:
        pd.DataFrame: Consolidated evaluation results across models.
    """
    all_metrics = []

    for root, _, files in os.walk(results_dir):
        for file in files:
            if file.endswith('_evaluation.txt'):
                file_path = os.path.join(root, file)
                try:
                    metrics = parse_evaluation_file(file_path)
                    metrics['Ticker'] = file.split('_')[0].upper()
                    metrics['Source'] = os.path.basename(os.path.dirname(root))  # e.g., lstm, arima, ensemble_reverse_rsme
                    all_metrics.append(metrics)
                except Exception as e:
                    print(f"❌ Failed to parse {file_path}: {e}")

    return pd.DataFrame(all_metrics)


# === Demo/Test ===
if __name__ == "__main__":
    # Sample usage/demo
    y_true = [100, 102, 105, 107, 110]
    y_pred = [101, 103, 104, 106, 111]
    results = evaluate_forecast(y_true, y_pred, model_name="DemoModel")
    print_evaluation(results)

    # Test loading results directory if exists
    results_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    if os.path.isdir(results_path):
        print("\n📂 Loading all evaluations from:", results_path)
        df = load_all_model_evaluations(results_path)
        print(df.head())
    else:
        print("\n⚠️  Results directory not found for evaluation aggregation.")
