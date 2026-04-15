Smart Burnout Predictor

📌 Overview
The Smart Burnout Predictor is an interactive machine learning web application built using Streamlit that predicts an employee’s burnout risk level (Low, Medium, High) based on daily work habits and behavioral patterns.
This project helps individuals and organizations proactively identify burnout risk and take preventive actions to improve productivity and well-being.

🚀 Features
🔮 Real-time Burnout Prediction
📊 Interactive Data Visualizations
📈 Model Performance Metrics (Accuracy & Confusion Matrix)
🎯 Feature Importance Analysis
🧠 Machine Learning powered (Random Forest)
🎛️ User-friendly dashboard with sliders and inputs


🛠️ Tech Stack
Frontend/UI: Streamlit
Backend: Python
Libraries Used:
    Pandas, NumPy
    Scikit-learn
    Imbalanced-learn (SMOTE)
    Matplotlib, Seaborn
    Plotly


📂 Dataset
Dataset used: work_from_home_burnout_dataset.csv
Contains features like:
    Work Hours
    Screen Time
    Meetings Count
    Breaks Taken
    Sleep Hours
    Task Completion Rate
    Burnout Score
    Day Type (Weekday/Weekend)

⚙️ How It Works
1. Data Preprocessing
Removed missing values and duplicates
Encoded categorical variables using Label Encoding
Balanced dataset using SMOTE
2. Model Training
Algorithm: Random Forest Classifier
Parameters:
n_estimators = 200
class_weight = balanced
Train-Test Split: 80% training, 20% testing
3. Prediction
User inputs work metrics via sidebar
Model predicts burnout risk:
🟢 Low
🟡 Medium
🔴 High
4. Output
Displays:
    Risk Level (color-coded)
    Accuracy Score
    Confusion Matrix

📊 Visualizations
Feature Importance (Key burnout drivers)
Work Hours vs Sleep Correlation
Burnout Risk Distribution

▶️ How to Run the Project
Step 1: Clone Repository
git clone https://github.com/your-username/burnout-predictor.git
cd burnout-predictor
Step 2: Install Dependencies
pip install -r requirements.txt
Step 3: Run the App
uv run streamlit run app.py

📈 Model Performance
The model achieves good accuracy using balanced data
Confusion Matrix is used to evaluate classification performance
Handles class imbalance effectively using SMOTE

🧠 Key Insights
Higher work hours + low sleep → Higher burnout risk
Frequent breaks reduce burnout probability
After-hours work significantly increases risk

⚠️ Limitations
Uses synthetic dataset (may not reflect real-world perfectly)
Limited features (mental health, stress levels not included)
Model can be improved with real-world data

🔮 Future Improvements
Use real-world datasets
Add deep learning models
Deploy on cloud (AWS / Azure / GCP)
Add user authentication & history tracking
Improve UI/UX design

👥 Team Contributions
Bindhu Sahithi – Model Training & Prediction Logic
Rishaniya Parthasarathy – Data Cleaning & Evaluation
Sharaban Tahura – UI Design & Styling
Lawrence Jaba Anand & Kevin Jeff Raj – Visualizations

📌 Conclusion
The Smart Burnout Predictor demonstrates how machine learning can be applied to workplace wellness. It provides actionable insights that help users maintain a healthy work-life balance and prevent burnout proactively.