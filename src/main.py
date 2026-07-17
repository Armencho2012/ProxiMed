import tkinter as tk
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from model import CustomKNN
import pandas as pd
import os 

class HealthApp(tb.Window):
    def __init__(self, knn_model):
        super().__init__(themename="superhero")
        self.knn_model = knn_model
        self.title("ProxiMed - Diabetes Risk Assessment Panel")
        self.geometry("1100x750")
        
        self.input_variables = {}
        self.binary_features = ["HighBP", "HighChol", "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "DiffWalk", "Sex"]
        
        self.build_interface()

    def build_interface(self):
        left_container = tb.Frame(self, padding=20)
        left_container.pack(side=LEFT, fill=BOTH, expand=True)
        
        form_label = tb.Label(left_container, text="Patient Questionnaire", font=("Helvetica", 18, "bold"), bootstyle="primary")
        form_label.pack(anchor=W, pady=(0, 20))

        scroll_canvas = tk.Canvas(left_container, bg="#2b3e50", highlightthickness=0)
        scrollbar = tb.Scrollbar(left_container, orient="vertical", command=scroll_canvas.yview)
        scroll_frame = tb.Frame(scroll_canvas)

        scroll_frame.bind("<Configure>", lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all")))
        scroll_canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        scroll_canvas.configure(yscrollcommand=scrollbar.set)

        scroll_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)

        for attr in self.knn_model.feature_columns:
            row = tb.Frame(scroll_frame, padding=5)
            row.pack(fill=X, pady=6)
            tb.Label(row, text=attr, width=25, font=("Helvetica", 11)).pack(side=LEFT)
            
            if attr in self.binary_features:
                var = tk.IntVar(value=0)
                tb.Radiobutton(row, text="Yes", variable=var, value=1, bootstyle="info").pack(side=LEFT, padx=10)
                tb.Radiobutton(row, text="No", variable=var, value=0, bootstyle="info").pack(side=LEFT)
                self.input_variables[attr] = var
            else:
                var = tk.StringVar(value="0")
                entry = tb.Entry(row, textvariable=var, bootstyle="info", width=15)
                entry.pack(side=LEFT, padx=10)
                self.input_variables[attr] = var

        right_container = tb.Frame(self, padding=30, width=420, bootstyle="dark")
        right_container.pack(side=RIGHT, fill=BOTH, expand=False)
        right_container.pack_propagate(False)

        title_label = tb.Label(right_container, text="Health Risk Scanner", font=("Helvetica", 22, "bold"), bootstyle="inverse-dark")
        title_label.pack(pady=(20, 5), anchor=CENTER)
        
        subtitle = tb.Label(right_container, text="Custom KNN Predictive Engine", font=("Helvetica", 10), bootstyle="muted")
        subtitle.pack(pady=(0, 30), anchor=CENTER)

        self.btn = tb.Button(right_container, text="Analyze Risk & Metrics", bootstyle="success-outline", command=self.execute_prediction)
        self.btn.pack(pady=20, ipadx=30, ipady=10, fill=X)

        self.result_container = tb.Frame(right_container, padding=20, bootstyle="secondary")
        self.result_container.pack(fill=BOTH, expand=True, pady=10)

        self.result_label = tb.Label(self.result_container, text="Provide patient values on the left panel and click 'Analyze Risk & Metrics' to generate diagnostics.", font=("Helvetica", 11), justify=LEFT, bootstyle="inverse-secondary", wraplength=320)
        self.result_label.pack(anchor=NW)

    def execute_prediction(self):
        try:
            patient_data = {attr: float(var.get()) for attr, var in self.input_variables.items()}
            self.result_label.config(text="Calculating custom metrics... Please wait.", bootstyle="inverse-secondary")
            self.update()
            
            results = self.knn_model.analyze_patient(patient_data)
            
            is_risk = "Risk" in results["Status"]
            status_color = "inverse-danger" if is_risk else "inverse-success"
            self.result_container.configure(bootstyle="danger" if is_risk else "success")
            
            display_text = (
                f"DIAGNOSTIC STATUS:\n{results['Status'].upper()}\n\n"
                f"--- KNN Evaluation Metrics ---\n\n"
                f"Accuracy:   {results['Accuracy']:.4f}\n"
                f"Precision:  {results['Precision']:.4f}\n"
                f"Recall:     {results['Recall']:.4f}\n"
                f"F1-Score:   {results['F1-Score']:.4f}\n"
                f"ROC-AUC:    {results['ROC-AUC']:.4f}"
            )
            
            self.result_label.config(text=display_text, bootstyle=status_color)
            
        except ValueError:
            self.result_container.configure(bootstyle="danger")
            self.result_label.config(text="Error: Ensure all numeric fields contain valid numbers.", bootstyle="inverse-danger")

if __name__ == "__main__":
    if os.getcwd() != os.path.dirname(os.path.abspath(__file__)):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv('../data/database.csv')
    model = CustomKNN(df, k_neighbors=5)
    app = HealthApp(model)
    app.mainloop()