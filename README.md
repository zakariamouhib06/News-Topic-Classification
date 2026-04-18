# News Topic Classification Pipeline

Welcome to the **News Topic Classification** project! This is an end-to-end Machine Learning pipeline that automatically reads a news article's text and classifies it into corresponding categories (such as *Business, Sci/Tech, Sports, or World News*).

## 🚀 Project Overview

The core objective of this project is to take raw news data and build a complete AI system. 
The system is divided into 4 major modules handled by different team members:
1. **Data Collection**: Sourcing and extracting raw news data.
2. **NLP Preprocessing**: Cleaning text, removing stop-words, and vectorizing natural language using TF-IDF.
3. **Model Building & Evaluation**: Training Machine Learning models (Logistic Regression, Naive Bayes, Linear SVM) and picking the best one based on F1-metrics.
4. **API & Interface**: Wrapping the trained model using FastAPI and providing an Electron/React Graphical User Interface (GUI) for end-users to interact with the classifier.

---

## 🛠️ How to Work with the Project

To run this pipeline locally on your machine, follow these steps:

### Step 1: Install Dependencies
Ensure you have Python installed. You must install the requested packages for the Machine Learning and Data processing logic.
```bash
pip install -r requirements.txt
```

### Step 2: Run the NLP Preprocessing
Before a model can be trained, the raw dataset must be cleaned and transformed into numerical features. 
Run the preprocessing script to generate the Text Vectorizer and Data Splits:
```bash
python preprocessing.py
```
*(This will generate `vectorizer.pkl` and data splits inside the dataset folder)*

### Step 3: Train & Save the ML Model
Once the data is preprocessed, launch the model training sequence. This script will train 3 different model algorithms, compare them, build a Confusion Matrix graph in the `reports/` folder, and save the ultimate best model dynamically:
```bash
python src/model_training.py
```
*(This will save `best_model.joblib` inside the `models/` directory)*

### Step 4: Test the Predictions
To verify that the ML model works before integrating it into FastAPI, you can use the predictor service directly. It will return the top category and class probabilities:
```bash
python src/predictor.py
```

### Step 5: (Upcoming) Launch FastAPI & Frontend
Once the backend team connects `predictor.py` to a FastAPI server, you will be able to launch `api/main.py` and run a Desktop Interface to query news articles interactively!
