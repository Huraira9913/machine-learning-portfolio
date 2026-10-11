# Machine Learning Portfolio

A curated collection of **end-to-end Machine Learning projects** demonstrating practical application of supervised and unsupervised algorithms. Each project includes data exploration, preprocessing, model training, evaluation, and a Streamlit demo app where applicable.

Inspired by comprehensive ML portfolios while using **original implementations** and public/safe datasets.

## Projects Overview

| # | Project | Algorithm | Type |
|---|---------|-----------|------|
| 01 | Salary Prediction | Linear Regression | Regression |
| 02 | Medical Insurance Cost | Linear Regression | Regression |
| 03 | SONAR Rock vs Mine | Logistic Regression | Classification |
| 04 | Fake News Detection | Logistic Regression + TF-IDF | NLP Classification |
| 05 | Credit Card Fraud Detection | Logistic Regression + SMOTE | Imbalanced Classification |
| 06 | Heart Disease Prediction | Logistic Regression | Classification |
| 07 | Titanic Survival | Logistic Regression | Classification |
| 08 | Spam Mail Detection | Logistic Regression + TF-IDF | NLP |
| 09 | Weather Prediction | Naive Bayes | Classification |
| 10 | SMS Spam Classifier | MultinomialNB | NLP |
| 11 | CPU Price Prediction | Decision Tree Regressor | Regression |
| 12 | Iris Classifier | K-Nearest Neighbors | Classification |
| 13 | Student Marks Prediction | KNN Regressor | Regression |
| 14 | Air Quality Prediction | KNN Regressor | Regression |
| 15 | Diabetes Prediction | Multiple models | Classification |
| 16 | Wine Quality | Random Forest | Classification/Regression |
| 17 | Telecom Customer Churn | Logistic + RF | Classification |
| 18 | Breast Cancer Detection | Logistic + SVM | Classification |
| 19 | House Price Prediction | Linear + RF | Regression |
| 20 | Mall Customer Segmentation | K-Means | Clustering |
| 21 | Calories Burnt Prediction | XGBoost / RF | Regression |

## Tech Stack

- **Python 3.10+**
- **Core**: Pandas, NumPy, scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Apps**: Streamlit
- **Extras**: imbalanced-learn, joblib

## How to Run

```bash
git clone https://github.com/Huraira9913/machine-learning-portfolio.git
cd machine-learning-portfolio
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run a specific script
python projects/01_salary_prediction/salary_prediction.py
python projects/05_fraud_detection/fraud_detection.py
python projects/12_iris_classifier/iris_knn.py
```

## Skills Developed

- End-to-end ML workflow (problem → data → model → evaluation → demo)
- Handling imbalanced data, text data, tabular data
- Proper train/test splits, cross-validation, metrics selection
- Packaging models with joblib and serving via Streamlit
- Clean, reproducible code structure

## Repository Structure

```
machine-learning-portfolio/
├── projects/
│   ├── 01_salary_prediction/
│   ├── 05_fraud_detection/
│   └── ...
├── requirements.txt
└── README.md
```

---

Built by **Muhammad AbuHuraira** as part of a professional Data Science portfolio.
