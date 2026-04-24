import os
import joblib

class PredictorService:
    def __init__(self, models_path=None):
        """
        Initializes the predictor by loading the saved models from disk.
        """
        if models_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.models_path = os.path.join(base_dir, 'models')
        else:
            self.models_path = models_path
        
        vectorizer_path = os.path.join(self.models_path, 'vectorizer.pkl')
        model_path = os.path.join(self.models_path, 'best_model.joblib')
        
        if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
            raise FileNotFoundError(
                "Model files not found. Ensure that both 'vectorizer.pkl' (from Maroine)"
                " and 'best_model.joblib' (from Salah) are in the 'models/' folder."
            )
            
        print("Loading Model and Vectorizer...")
        self.vectorizer = joblib.load(vectorizer_path)
        self.model = joblib.load(model_path)
        
        # Determine the classes the model learned
        if hasattr(self.model, 'classes_'):
            self.classes = self.model.classes_
        else:
            self.classes = []

    def predict(self, text: str):
        """
        Takes raw article text, applies preprocessing (vectorizer), and predicts the category.
        Returns the top predicted label and the class probabilities.
        """
        if not text or len(text.strip()) == 0:
            return {"error": "Input text is strictly empty."}
            
        # 1. Pipeline step: Transform raw text into features using Maroine's Vectorizer
        features = self.vectorizer.transform([text])
        
        # 2. Pipeline step: Predict using Salah's best model
        predicted_label = self.model.predict(features)[0]
        
        response = {
            "predicted_category": str(predicted_label)
        }
        
        # 3. Add probabilities if the model supports it
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)[0]
            
            # Map probabilities to class names
            class_probs = {
                str(cls_name): round(float(prob), 4) 
                for cls_name, prob in zip(self.classes, probabilities)
            }
            
            # Optionally sort by highest probability
            class_probs = dict(sorted(class_probs.items(), key=lambda item: item[1], reverse=True))
            response["probabilities"] = class_probs
            
        return response

if __name__ == "__main__":
    # Small test script to verify it works
    print("Testing PredictorService...")
    try:
        service = PredictorService()
        sample_text = "The new graphics card provides amazing fps in games."
        print(f"Input: {sample_text}")
        result = service.predict(sample_text)
        print(f"Output: {result}")
    except Exception as e:
        print(f"Error: {e}")
