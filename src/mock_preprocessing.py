import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def run_mock_preprocessing():
    print("Running Mock Preprocessing (Simulating Maroine's Job)...")
    
    # 1. Create a dummy dataset
    data = {
        "text": [
            "Apple releases a new iPhone with better battery.",
            "The stock market crashed today reflecting tech decline.",
            "Real Madrid wins the champion league after intense match.",
            "A new AI algorithm accurately predicts weather changes.",
            "Local elections resulted in a surprising new mayor.",
            "The basketball team scored 100 points in one quarter.",
            "New regulations on banking passed by the congress.",
            "Movie director wins best picture award at the Oscars."
        ] * 10,
        "label": [
            "Technology", "Business", "Sports", "Technology", 
            "Politics", "Sports", "Politics", "Entertainment"
        ] * 10
    }
    
    df = pd.DataFrame(data)
    
    # 2. Fit a Vectorizer (Maroine's job)
    vectorizer = TfidfVectorizer(stop_words='english')
    X_transformed = vectorizer.fit_transform(df['text'])
    y = df['label']
    
    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.25, random_state=42)
    
    # 4. Save artifacts
    models_dir = 'models'
    processed_dir = 'data/processed'
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save the vectorizer
    joblib.dump(vectorizer, os.path.join(models_dir, 'vectorizer.joblib'))
    
    # Save datasets (Using joblib for sparse matrices which is better than csv)
    joblib.dump((X_train, y_train), os.path.join(processed_dir, 'train_data.joblib'))
    joblib.dump((X_test, y_test), os.path.join(processed_dir, 'test_data.joblib'))
    
    print("Mock preprocessing finished!")
    print("Saved -> models/vectorizer.joblib")
    print("Saved -> data/processed/train_data.joblib & test_data.joblib\n")

if __name__ == "__main__":
    run_mock_preprocessing()
