import pandas as pd
import math
import random

class CustomKNN:
    def __init__(self, data_frame, k_neighbors=5):
        self.data_frame = data_frame
        self.k_neighbors = k_neighbors
        self.feature_columns = [col for col in self.data_frame.columns if col not in ['Diabetes_binary', 'CholCheck', 'AnyHealthcare', 'NoDocbcCost']]
        self.features = self.data_frame[self.feature_columns].values.tolist()
        self.labels = self.data_frame['Diabetes_binary'].values.tolist()
        self.min_max_values = {}
        self.normalize_dataset()

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
        test_indices = random.sample(range(len(self.features)), min(test_sample_size, len(self.features)))
        true_labels = []
        predicted_probs = []

        for index in test_indices:
            test_vector = self.features[index]
            actual_label = self.labels[index]
            
            distances = []
            for train_index in range(len(self.features)):
                if train_index == index:
                    continue
                dist = self.calculate_distance(test_vector, self.features[train_index])
                distances.append((dist, self.labels[train_index]))
            
            distances.sort(key=lambda x: x[0])
            nearest_neighbors = [label for _, label in distances[:self.k_neighbors]]
            positive_votes = nearest_neighbors.count(1.0)
            probability = positive_votes / self.k_neighbors
            
            true_labels.append(actual_label)
            predicted_probs.append(probability)

        tp = sum(1 for t, p in zip(true_labels, predicted_probs) if p >= 0.5 and t == 1.0)
        tn = sum(1 for t, p in zip(true_labels, predicted_probs) if p < 0.5 and t == 0.0)
        fp = sum(1 for t, p in zip(true_labels, predicted_probs) if p >= 0.5 and t == 0.0)
        fn = sum(1 for t, p in zip(true_labels, predicted_probs) if p < 0.5 and t == 1.0)

        accuracy = (tp + tn) / len(true_labels) if len(true_labels) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

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

        return accuracy, precision, recall, f1_score, roc_auc

    def analyze_patient(self, user_input_dict):
        normalized_target = self.normalize_input(user_input_dict)
        distances = []
        for index, train_vector in enumerate(self.features):
            dist = self.calculate_distance(normalized_target, train_vector)
            distances.append((dist, self.labels[index]))
        
        distances.sort(key=lambda x: x[0])
        nearest_neighbors = [label for _, label in distances[:self.k_neighbors]]
        positive_votes = nearest_neighbors.count(1.0)
        
        is_diabetic = positive_votes >= (self.k_neighbors / 2)
        status = "Diabetic / Prediabetic Risk" if is_diabetic else "Healthy"
        
        acc, prec, rec, f1, auc = self.compute_metrics(test_sample_size=50)
        
        return {
            "Status": status,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc
        }