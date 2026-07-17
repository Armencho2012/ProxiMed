import pandas as pd
import math
import random
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

class CustomKNN:
    def __init__(self, data_frame, k_neighbors=5):
        self.data_frame = data_frame
        self.k_neighbors = k_neighbors
        self.feature_columns = [col for col in self.data_frame.columns if col not in ['Diabetes_binary', 'CholCheck', 'AnyHealthcare', 'NoDocbcCost']]
        self.features = self.data_frame[self.feature_columns].values.tolist()
        self.labels = self.data_frame['Diabetes_binary'].values.tolist()
        self.min_max_values = {}
        self.normalize_dataset()
        
        # 1. Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.labels, test_size=0.2, random_state=42, stratify=self.labels
        )
        
        # 2. Apply SMOTE only to training set
        smote = SMOTE(random_state=42)
        self.X_train_res, self.y_train_res = smote.fit_resample(self.X_train, self.y_train)
        
        # Vectorized training data for rapid distance calculations
        self.X_train_res_np = np.array(self.X_train_res)
        self.y_train_res_np = np.array(self.y_train_res)

    def normalize_dataset(self):
        for col_index, col_name in enumerate(self.feature_columns):
            col_data = [row[col_index] for row in self.features]
            min_val = min(col_data)
            max_val = max(col_data)
            self.min_max_values[col_name] = (min_val, max_val)
            if max_val > min_val:
                for row_index in range(len(self.features)):
                    self.features[row_index][col_index] = (self.features[row_index][col_index] - min_val) / (max_val - min_val)

    def normalize_input(self, user_input_dict):
        normalized_vector = []
        for col_name in self.feature_columns:
            min_val, max_val = self.min_max_values[col_name]
            raw_val = user_input_dict[col_name]
            if max_val > min_val:
                normalized_vector.append((raw_val - min_val) / (max_val - min_val))
            else:
                normalized_vector.append(0.0)
        return normalized_vector

    def calculate_distance(self, vector_a, vector_b):
        squared_differences = [(a - b) ** 2 for a, b in zip(vector_a, vector_b)]
        return math.sqrt(sum(squared_differences))

    def compute_metrics(self, test_sample_size=50):
        # Sample test_sample_size indices from X_test (untouched)
        test_indices = random.sample(range(len(self.X_test)), min(test_sample_size, len(self.X_test)))
        true_labels = []
        predicted_probs = []

        for index in test_indices:
            test_vector = np.array(self.X_test[index])
            actual_label = self.y_test[index]
            
            # Vectorized Euclidean distance squared (sqrt is not needed for sorting neighbors)
            diff = self.X_train_res_np - test_vector
            dist_sq = np.sum(diff ** 2, axis=1)
            
            # Fast get of top k indices
            nearest_indices = np.argsort(dist_sq)[:self.k_neighbors]
            nearest_neighbors = self.y_train_res_np[nearest_indices]
            
            positive_votes = np.sum(nearest_neighbors == 1.0)
            probability = positive_votes / self.k_neighbors
            
            true_labels.append(actual_label)
            predicted_probs.append(probability)

        # 0.5 Threshold Metrics
        tp_50 = sum(1 for t, p in zip(true_labels, predicted_probs) if p >= 0.5 and t == 1.0)
        tn_50 = sum(1 for t, p in zip(true_labels, predicted_probs) if p < 0.5 and t == 0.0)
        fp_50 = sum(1 for t, p in zip(true_labels, predicted_probs) if p >= 0.5 and t == 0.0)
        fn_50 = sum(1 for t, p in zip(true_labels, predicted_probs) if p < 0.5 and t == 1.0)

        accuracy_50 = (tp_50 + tn_50) / len(true_labels) if len(true_labels) > 0 else 0.0
        precision_50 = tp_50 / (tp_50 + fp_50) if (tp_50 + fp_50) > 0 else 0.0
        recall_50 = tp_50 / (tp_50 + fn_50) if (tp_50 + fn_50) > 0 else 0.0
        f1_score_50 = 2 * (precision_50 * recall_50) / (precision_50 + recall_50) if (precision_50 + recall_50) > 0 else 0.0

        # 0.3 Threshold Metrics (Tuned)
        tp_30 = sum(1 for t, p in zip(true_labels, predicted_probs) if p >= 0.3 and t == 1.0)
        tn_30 = sum(1 for t, p in zip(true_labels, predicted_probs) if p < 0.3 and t == 0.0)
        fp_30 = sum(1 for t, p in zip(true_labels, predicted_probs) if p >= 0.3 and t == 0.0)
        fn_30 = sum(1 for t, p in zip(true_labels, predicted_probs) if p < 0.3 and t == 1.0)

        accuracy_30 = (tp_30 + tn_30) / len(true_labels) if len(true_labels) > 0 else 0.0
        precision_30 = tp_30 / (tp_30 + fp_30) if (tp_30 + fp_30) > 0 else 0.0
        recall_30 = tp_30 / (tp_30 + fn_30) if (tp_30 + fn_30) > 0 else 0.0
        f1_score_30 = 2 * (precision_30 * recall_30) / (precision_30 + recall_30) if (precision_30 + recall_30) > 0 else 0.0

        thresholds = sorted(list(set(predicted_probs)), reverse=True)
        tpr_list = [0.0]
        fpr_list = [0.0]
        for thresh in thresholds:
            t_tp = sum(1 for t, p in zip(true_labels, predicted_probs) if p >= thresh and t == 1.0)
            t_fp = sum(1 for t, p in zip(true_labels, predicted_probs) if p >= thresh and t == 0.0)
            t_tn = sum(1 for t, p in zip(true_labels, predicted_probs) if p < thresh and t == 0.0)
            t_fn = sum(1 for t, p in zip(true_labels, predicted_probs) if p < thresh and t == 1.0)
            
            current_tpr = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0.0
            current_fpr = t_fp / (t_fp + t_tn) if (t_fp + t_tn) > 0 else 0.0
            tpr_list.append(current_tpr)
            fpr_list.append(current_fpr)

        roc_auc = 0.0
        for i in range(1, len(tpr_list)):
            roc_auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2

        return {
            "Accuracy_50": accuracy_50,
            "Precision_50": precision_50,
            "Recall_50": recall_50,
            "F1-Score_50": f1_score_50,
            "Accuracy_30": accuracy_30,
            "Precision_30": precision_30,
            "Recall_30": recall_30,
            "F1-Score_30": f1_score_30,
            "ROC-AUC": roc_auc
        }

    def analyze_patient(self, user_input_dict):
        normalized_target = np.array(self.normalize_input(user_input_dict))
        
        # Vectorized Euclidean distance squared
        diff = self.X_train_res_np - normalized_target
        dist_sq = np.sum(diff ** 2, axis=1)
        
        nearest_indices = np.argsort(dist_sq)[:self.k_neighbors]
        nearest_neighbors = self.y_train_res_np[nearest_indices]
        
        positive_votes = np.sum(nearest_neighbors == 1.0)
        probability = positive_votes / self.k_neighbors
        
        is_diabetic = probability >= 0.3
        status = "Diabetic / Prediabetic Risk" if is_diabetic else "Healthy"
        
        metrics = self.compute_metrics(test_sample_size=50)
        
        return {
            "Status": status,
            "Accuracy_50": metrics["Accuracy_50"],
            "Precision_50": metrics["Precision_50"],
            "Recall_50": metrics["Recall_50"],
            "F1-Score_50": metrics["F1-Score_50"],
            "Accuracy_30": metrics["Accuracy_30"],
            "Precision_30": metrics["Precision_30"],
            "Recall_30": metrics["Recall_30"],
            "F1-Score_30": metrics["F1-Score_30"],
            "ROC-AUC": metrics["ROC-AUC"]
        }