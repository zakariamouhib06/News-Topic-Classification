# News Topic Classification Pipeline

Welcome to the **News Topic Classification** project! This is an end-to-end Machine Learning ecosystem that automatically analyzes a news article's text and classifies it into one of four primary categories using advanced Natural Language Processing (NLP). 

The app includes an integrated, ultra-modern Desktop UI powered by React, Electron, and a FastAPI machine learning backend.

## 🚀 Project Overview

The core objective of this project is to take raw news data and build a sophisticated AI application encompassing everything from data processing to a premium Graphical User Interface (GUI). 

### 1. NLP Preprocessing
Raw dataset text is cleaned (removing stop words, standardizing formats) and transformed into numerical features using a TF-IDF text vectorizer (`vectorizer.pkl`).

### 2. Model Building & Evaluation
The training sequence fits and compares three classical machine learning models:
- Naive Bayes
- Logistic Regression
- Linear Support Vector Machines (SVM)

After evaluating accuracy, precision, recall, and F1-score, the script dynamically saves the most performant model (`best_model.joblib`) for production inference. Our trained model currently utilizes **Logistic Regression** achieving a ~91% F1-score across all categories.

### 3. FastAPI Backend
The best model is loaded into memory, wrapped inside `predictor.py`, and exposed through a lightweight **FastAPI** server that enables HTTP-based semantic predictions.

### 4. Modern Desktop Interface
A visually stunning, **Glassmorphism-styled** Desktop Application built using **React** (via Vite) and wrapped natively with **Electron**. The UI is fully responsive, interactive, and features dynamic animations and glowing design tokens.

---

## 📊 Recognizable Categories

The machine learning pipeline has been trained specifically to predict text belonging to the following categories:
1. **Business**
2. **Sci/Tech** (Science and Technology)
3. **Sports**
4. **World** (World News - including general Politics)

*Note: Political articles are generally classified under "World" news.*

---

## 🛠️ Setup & Installation Instructions

To run this complete pipeline locally, follow these simple steps:

### Step 1: Install Python Dependencies
For the Machine Learning and Backend API logic:
```bash
pip install -r requirements.txt
```

### Step 2: Run the Machine Learning Pipeline (One-Time)
Before launching the app, you must train the model so the API has something to serve! 
Run these commands in order from the root directory:

**1. Transform Data & Generate Vectorizer:**
```bash
python preprocessing.py
```

**2. Train the Best ML Model:**
```bash
python src/model_training.py
```

### Step 3: Install Frontend Dependencies
Navigate to the `desktop-app` folder and install NPM packages:
```bash
cd desktop-app
npm install
```

---

## 🔥 Running the Full Application

We've configured a robust, unified start process that runs the frontend, backend, and desktop wrapper concurrently all from a single command!

Open your terminal, navigate to the `desktop-app` directory, and run:
```bash
npm run electron:dev
```

**What this does:**
1. Spins up the FastAPI Backend via Uvicorn (`http://localhost:8000`).
2. Starts the Vite React development server for hot-reloading (`http://localhost:5173`).
3. Launches the native **Electron Desktop Application**.

You simply paste your text into the UI, click "Classify Article", and the app will smoothly animate and present the prediction logic directly from the ML backend!
