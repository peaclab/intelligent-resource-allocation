import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def random_color():
    return np.random.rand(3,)

def plot_raw_results(y_pred, y_test, req_test, target_feature, df_name, filepath):
    color_actual_vs_requested = random_color()
    color_actual_vs_predicted = random_color()
    color_predicted_vs_actual = random_color()

    plt.figure(figsize=(14, 10))

    # Subplot 1: y_test vs req_test and y_pred
    plt.subplot(2, 1, 1)
    plt.scatter(y_test, req_test, label='User Requested Values', color=color_actual_vs_requested, alpha=0.7)
    plt.scatter(y_test, y_pred, label='XGBoost Predicted Values', color=color_actual_vs_predicted, alpha=0.7, marker='x')
    plt.plot(y_test, y_test, label='Perfect Prediction Line', color='red', linestyle='--', alpha=1)
    plt.xlabel('Actual')
    plt.ylabel('Requested/Predicted')
    plt.title(f'Comparison of Requested and Predicted {target_feature} on {df_name}')
    plt.legend()
    plt.grid(True)

    # Subplot 2: y_pred vs y_test
    plt.subplot(2, 1, 2)
    plt.plot(y_test, label='Actual', color='blue', alpha=0.7)
    plt.plot(y_pred, label='Predicted', color='orange', alpha=0.7)
    plt.xlabel('Job Index')
    plt.ylabel('Value')
    plt.title(f'Prediction Performance of {target_feature} on {df_name}')
    plt.legend()
    # Display grid
    plt.grid(True)

    # Adjust layout
    plt.tight_layout()

    plot_name = 'over_under_prediction_analysis.png'
    file_path = os.path.join(filepath, plot_name)
    plt.savefig(file_path)

    plt.show()


def plot_kde_results(req_oe_ratio, pred_oe_ratio, target_feature, df_name):

    color_actual_vs_requested = random_color()
    color_actual_vs_predicted = random_color()


    plt.figure(figsize=(10, 6))

    sns.kdeplot(req_oe_ratio, label='Requested Value/Actual Usage', color=color_actual_vs_requested, fill=True, alpha=0.6, linewidth=2, log_scale=True)
    sns.kdeplot(pred_oe_ratio, label='Predicted/Actual Usage', color=color_actual_vs_predicted, fill=True, alpha=0.6, linewidth=2, log_scale=True)
    plt.axvline(1, color='red', linestyle='--', label='Perfect Match (Ratio=1)')

    plt.xlabel('Overestimation Ratio (Log Scale)', fontsize=14)
    plt.ylabel('Density')
    plt.title(f'Overestimation KDE Plot for {target_feature} on {df_name}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.xticks(fontsize=14)
    plt.show()

def plot_everything(metrics_data,overprediction_sums, title1, title2, ylabel, directory):
    
    metrics_df = pd.DataFrame(metrics_data, columns=['Cluster', 'Total Jobs', 'Requested > Actual', 'Predicted < Actual', 'Predicted > Actual'])

    overprediction_df = pd.DataFrame(overprediction_sums, columns=['Sum Predicted - Actual', 'Sum Requested - Actual'])
    overprediction_df['Cluster'] = range(1, len(overprediction_sums) + 1)

    #FIRST PLOT
    plt.figure(figsize=(12, 8))
    bar_width = 0.2
    x = np.arange(len(metrics_df))

    total_jobs = metrics_df['Total Jobs']
    percentages_predicted_less = metrics_df['Predicted < Actual'] / total_jobs * 100


    for idx, metric in enumerate(['Total Jobs', 'Requested > Actual', 'Predicted < Actual', 'Predicted > Actual']):
        bars = plt.bar(x + idx * bar_width, metrics_df[metric], width=bar_width, label=metric)
        if metric in ['Predicted < Actual']:
            for i, (bar, percentage) in enumerate(zip(bars, percentages_predicted_less)):
                plt.text(
                    bar.get_x() + bar.get_width() / 2, 
                    bar.get_height() + 1,
                    f"{percentage:.1f}%", 
                    ha='center', va='bottom', fontsize=12, color='black' )

    plt.xlabel('Cluster', fontsize=14)
    plt.ylabel('Number of Jobs', fontsize=14)
    plt.title(f'{title1}', fontsize=16)
    plt.xticks(x + 1.5 * bar_width, metrics_df['Cluster'], fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plot_name = 'over_under_prediction_analysis.png'
    file_path = os.path.join(directory, plot_name)
    plt.savefig(file_path)

    plt.show()

    # SECOND PLOT
    plt.figure(figsize=(14, 8))
    x = np.arange(len(overprediction_df)) 
    bar_width = 0.4

    plt.bar(x - bar_width / 2, overprediction_df['Sum Requested - Actual'] , width=bar_width, label='Sum of all (Requested - Actual)', color='purple')
    plt.bar(x + bar_width / 2, overprediction_df['Sum Predicted - Actual'] , width=bar_width, label='Sum of all (Predicted - Actual)', color='orange')

    plt.xlabel('Cluster', fontsize=14)
    plt.ylabel(f'Sum of Differences in {ylabel} (Log Scale)', fontsize=14)
    plt.title(f'{title2}', fontsize=16)
    plt.xticks(x, overprediction_df['Cluster'], fontsize=14)
    plt.yscale('log') 
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plot_name = 'sum_of_differences.png'
    file_path = os.path.join(directory, plot_name)
    plt.savefig(file_path)

    plt.show()




