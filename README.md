Credit Card Fraud Detection using Machine Learning
Project Overview

This project aims to detect fraudulent credit card transactions using machine learning algorithms. The system analyzes historical transaction data to classify whether a transaction is fraudulent or legitimate. It can help banks and financial institutions prevent financial losses and secure user accounts.

Features

Train machine learning models on credit card transaction data.

Detect fraudulent transactions in real-time or batch mode.

Provides evaluation metrics to assess model performance.

Supports CSV input for testing and training datasets.

Dataset

The dataset used is a credit card transactions dataset (like the Kaggle Credit Card Fraud Detection Dataset
).

Dataset contains anonymized features (V1, V2, ..., V28), Time, Amount, and a Class label:

Class = 0 → Legitimate transaction

Class = 1 → Fraudulent transaction



Installation

Clone the repository:

git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection


Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows


Install dependencies:

pip install -r requirements.txt

Requirements

Python 3.8+

Pandas

NumPy

Scikit-learn

Matplotlib / Seaborn (for visualization)

Joblib (for saving trained models)
