# tools/compare_models.py

import os
import glob
import pandas as pd
from tabulate import tabulate
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIG ===
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
MODELS = ['arima', 'lstm', 'prophet', 'xgboost', 'ensemble_reverse_rsme']
OUTPUT_CSV = os.path.join(RESULTS_DIR, 'model_comparison.csv')
PLOT_DIR = os.path.join(RESULTS_DIR, 'comparison_charts')
WEB_DASHBOARD_DIR = os.path.join(RESULTS_DIR, 'dashboard')
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(WEB_DASHBOARD_DIR, exist_ok=True)

# === FUNCTIONS ===
def load_evaluation_file(path, model):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        data = {'Model': model}
        for line in lines:
            if ':' in line:
                k, v = line.strip().split(':', 1)
                k = k.strip().upper()
                try:
                    data[k] = float(v.strip())
                except ValueError:
                    continue
        return data
    except:
        return None

def aggregate_results():
    records = []
    for model in MODELS:
        model_dir = os.path.join(RESULTS_DIR, model)
        if not os.path.exists(model_dir):
            continue
        for eval_file in glob.glob(f'{model_dir}/*/*_evaluation.txt'):
            ticker = os.path.basename(eval_file).split('_')[0].upper()
            result = load_evaluation_file(eval_file, model.upper())
            if result:
                result['Ticker'] = ticker
                records.append(result)
    df = pd.DataFrame(records)
    return df[['Ticker', 'Model', 'MAE', 'MSE', 'RMSE']]

def get_best_models(df):
    return df.loc[df.groupby('Ticker')['RMSE'].idxmin()].sort_values(by='RMSE')

def plot_comparison(df):
    for ticker in df['Ticker'].unique():
        subset = df[df['Ticker'] == ticker].sort_values(by='RMSE')
        plt.figure(figsize=(10, 5))
        sns.barplot(x='Model', y='RMSE', data=subset, palette='viridis')
        plt.title(f'RMSE Comparison for {ticker}')
        plt.ylabel('RMSE')
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_path = os.path.join(PLOT_DIR, f'{ticker.lower()}_rmse_comparison.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"📊 Chart saved to {plot_path}")

def save_web_dashboard(df_best):
    html_path = os.path.join(WEB_DASHBOARD_DIR, 'index.html')
    table_html = df_best.to_html(index=False, classes='table table-striped')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(f"""
        <html>
        <head>
            <title>Model Comparison Dashboard</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
        </head>
        <body>
            <div class="container mt-5">
                <h2>Best Performing Models by Ticker</h2>
                {table_html}
            </div>
        </body>
        </html>
        """)
    print(f"🌐 Web dashboard saved to {html_path}")

# === MAIN ===
def main():
    print("📊 Aggregating model evaluations...")
    df_all = aggregate_results()
    df_all.to_csv(OUTPUT_CSV, index=False)
    print(tabulate(df_all, headers='keys', tablefmt='grid'))

    print("🏆 Filtering best models by RMSE...")
    df_best = get_best_models(df_all)
    print(tabulate(df_best, headers='keys', tablefmt='grid'))

    print("📈 Generating comparison plots...")
    plot_comparison(df_all)

    print("🌐 Exporting web dashboard...")
    save_web_dashboard(df_best)

if __name__ == '__main__':
    main()
