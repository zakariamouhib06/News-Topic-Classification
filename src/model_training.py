import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

class ModelTrainer:
    def __init__(self, data_path=None, models_path=None, reports_path=None):
        """
        Initializes the ModelTrainer.iop
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = data_path or os.path.join(base_dir, 'data', 'processed')
        self.models_path = models_path or os.path.join(base_dir, 'models')
        self.reports_path = reports_path or os.path.join(base_dir, 'reports')
        
        # We wrap LinearSVC with CalibratedClassifierCV so it can predict probabilities (predict_proba)
        # which is often required by frontend/API to show "confidence score".
        self.models = {
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Linear SVM': CalibratedClassifierCV(LinearSVC(dual="auto", max_iter=2000))
        }
        
        self.best_model_name = None
        self.best_model = None
        
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.reports_path, exist_ok=True)

    def load_data(self):
        """Loads training and testing data exported by Maroine's preprocessing module."""
        print("Loading processed data...")
        self.X_train, self.X_test, self.y_train, self.y_test = joblib.load(f"{self.data_path}/splits.pkl")
        
    def train_and_evaluate(self):
        """Trains candidate models and evaluates them to find the best one."""
        print("\nStarting Model Training...")
        results = []
        
        best_f1 = 0
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            # Train
            model.fit(self.X_train, self.y_train)
            
            # Predict
            y_pred = model.predict(self.X_test)
            
            # Metrics
            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            
            results.append({
                'Model': name,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1-Score': f1
            })
            
            # Select Best Model based on F1-Score
            if f1 >= best_f1:
                best_f1 = f1
                self.best_model_name = name
                self.best_model = model
                
        self.generate_report(results)
    
    def generate_report(self, results):
        """Saves evaluation metrics and confusion matrix of the best model."""
        print("\n--- Evaluation Report ---")
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        
        # Detailed report for the best model
        print(f"\nBest Model Selected: {self.best_model_name}")
        
        y_pred = self.best_model.predict(self.X_test)
        print("\nClassification Report (Best Model):")
        print(classification_report(self.y_test, y_pred, zero_division=0))
        
        # Generate Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Attempt to get labels ordered correctly
        labels = sorted(list(set(self.y_test)))
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix: {self.best_model_name}')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        
        cm_path = os.path.join(self.reports_path, 'confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()
        print(f"Saved confusion matrix graph to {cm_path}")
        
    def save_best_model(self):
        """Saves the best model for API prediction usage."""
        model_filepath = os.path.join(self.models_path, 'best_model.joblib')
        joblib.dump(self.best_model, model_filepath)
        print(f"Saved best model ({self.best_model_name}) to {model_filepath}")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.load_data()
    trainer.train_and_evaluate()
    trainer.save_best_model()
