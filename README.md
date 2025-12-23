🎴 Credit Card Fraud Detection using Machine Learning

📝 Project Overview

This project detects fraudulent credit card transactions using machine learning. By analyzing historical transaction data, the system classifies transactions as either fraudulent or legitimate.
It helps banks and financial institutions prevent financial losses and protect users.

✨ Features

Train machine learning models on credit card transaction data.

Detect fraudulent transactions in real-time or batch mode.

Provides evaluation metrics to assess model performance.

Supports CSV input for both training and testing datasets.

📊 Dataset

We use a credit card transactions dataset (like the Kaggle Credit Card Fraud Detection Dataset
).

Columns:

V1, V2, ..., V28 → anonymized features

Time → seconds elapsed between this transaction and the first transaction in the dataset

Amount → transaction amount

Class → target label (0 = Legitimate, 1 = Fraudulent)

Example:

Time	V1	V2	...	V28	Amount	Class
0	-1.36	-0.33	...	-0.44	149.62	0
1	1.12	0.23	...	0.12	2.69	1
🛠 Installation

1️⃣ Clone the repository:

git clone https://github.com/parameshwar21/credit-card-fraud-detection.git
cd credit-card-fraud-detection


2️⃣ Create a virtual environment (optional but recommended):

python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate


3️⃣ Install dependencies:

pip install -r requirements.txt

⚙ Requirements

Python 3.8+

Pandas

NumPy

Scikit-learn

Matplotlib / Seaborn (for visualization)

Joblib (for saving trained models)
