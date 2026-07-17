import pandas as pd
import os
import sys

# Ensure current folder is src
if os.getcwd() != os.path.dirname(os.path.abspath(__file__)):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

from model import CustomKNN

data = pd.read_csv('../data/database.csv')
model = CustomKNN(data)

def test_robot(values):
    attr_map = dict(zip(model.feature_columns, values))
    res = model.analyze_patient(attr_map)
    return res

# Automated test case
sample_input = [1.0, 1.0, 30.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 6.0, 6.0, 8.0]
res = test_robot(sample_input)
print(f"Robot Prediction Status (Tuned Threshold 0.3): {res['Status']}")

# Get metrics
metrics = model.compute_metrics(test_sample_size=50)

# Print before/after comparison table
print("\n" + "="*60)
print("             BEFORE / AFTER METRIC COMPARISON")
print("="*60)
print(f"{'Metric':<15} | {'Original':<10} | {'Tuned (0.3)':<12} | {'Default (0.5)':<12}")
print("-"*60)
print(f"{'Accuracy':<15} | {'0.9000':<10} | {metrics['Accuracy_30']:<12.4f} | {metrics['Accuracy_50']:<12.4f}")
print(f"{'Precision':<15} | {'0.5000':<10} | {metrics['Precision_30']:<12.4f} | {metrics['Precision_50']:<12.4f}")
print(f"{'Recall':<15} | {'0.2000':<10} | {metrics['Recall_30']:<12.4f} | {metrics['Recall_50']:<12.4f}")
print(f"{'F1-Score':<15} | {'0.2857':<10} | {metrics['F1-Score_30']:<12.4f} | {metrics['F1-Score_50']:<12.4f}")
print(f"{'ROC-AUC':<15} | {'0.7267':<10} | {metrics['ROC-AUC']:<12.4f} | {metrics['ROC-AUC']:<12.4f}")
print("="*60)