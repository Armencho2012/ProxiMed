import pandas as pd
import os
import sys
import random

script_dir = os.path.dirname(os.path.abspath(__file__))
if os.getcwd() != script_dir:
    os.chdir(script_dir)

from model import CustomKNN

csv_path = os.path.join('..', 'data', 'database.csv')
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Could not find dataset at {os.path.abspath(csv_path)}")

print("Loading dataset...")
data = pd.read_csv(csv_path)
print(f"Loaded dataset: {data.shape[0]} rows, {data.shape[1]} columns")

print("Initializing CustomKNN model...")
model = CustomKNN(data)
print(f"Feature columns used by model: {model.feature_columns}")

def run_bulk_test(n_tests=200, seed=42):
    random.seed(seed)
    n_tests = min(n_tests, len(data))
    sample_df = data.sample(n=n_tests, random_state=seed).reset_index(drop=True)
    
    correct = 0
    wrong = 0
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    results = []

    print("\nStarting bulk test execution...")
    for i, row in sample_df.iterrows():
        print(f"Running test {i + 1}/{n_tests}...", end="", flush=True)
        
        patient_values = row[model.feature_columns].tolist()
        attr_map = dict(zip(model.feature_columns, patient_values))
        
        res = model.analyze_patient(attr_map)
        
        predicted_status = res['Status']
        actual_label = row['Diabetes_binary']
        actual_status = "Diabetic / Prediabetic Risk" if actual_label == 1 else "Healthy"
        
        is_correct = (predicted_status == actual_status)
        correct += is_correct
        wrong += not is_correct
        
        predicted_positive = (predicted_status == "Diabetic / Prediabetic Risk")
        actual_positive = (actual_label == 1)
        
        if predicted_positive and actual_positive:
            true_positives += 1
        elif predicted_positive and not actual_positive:
            false_positives += 1
        elif not predicted_positive and actual_positive:
            false_negatives += 1
        else:
            true_negatives += 1
            
        results.append((i + 1, predicted_status, actual_status, is_correct))
        print(" Done.", flush=True)

    print("\n" + "=" * 70)
    print(f" BULK TEST RESULTS ({n_tests} real patients)")
    print("=" * 70)
    
    for num, pred, actual, ok in results:
        mark = "✓" if ok else "✗"
        print(f"Test {num:>3} | Predicted: {pred:<28} | Actual: {actual:<28} | {mark}")
        
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print(f"Total tests: {n_tests}")
    print(f"Correct: {correct} ({correct/n_tests*100:.2f}%)")
    print(f"Wrong: {wrong} ({wrong/n_tests*100:.2f}%)")
    print("-" * 70)
    print("Confusion Matrix breakdown:")
    print(f" True Positives (correctly flagged diabetic): {true_positives}")
    print(f" True Negatives (correctly flagged healthy): {true_negatives}")
    print(f" False Positives (healthy wrongly flagged risky): {false_positives}")
    print(f" False Negatives (diabetic missed as healthy): {false_negatives}")
    print("=" * 70)

if __name__ == "__main__":
    run_bulk_test(n_tests=200)