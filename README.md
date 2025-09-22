# 📊 Customer Churn Prediction

An **end-to-end machine learning project** for predicting customer churn in the telecom industry. This project is based on the [Telecom Churn Dataset](https://www.kaggle.com/datasets/barun2104/telecom-churn).

For a detailed explanation of every step in the pipeline, see [PIPELINE.md](PIPELINE.md).


## 🚀 Project Overview

* **Goal:** Predict whether a telecom customer will churn (leave the service).
* **Dataset:** Customer demographics, account information, and service usage patterns.
* **Workflow:**

  1. Exploratory Data Analysis (EDA)
  2. Data Preprocessing (pipelines with scikit-learn)
  3. Model Training & Evaluation
  4. Hyperparameter Tuning (RandomizedSearchCV)
  5. Feature Importance Analysis
  6. Artifact Export (preprocessor, model, pipeline)
  7. Inference on New Data


## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Libraries:**

  * `pandas`, `numpy`, `matplotlib`
  * `scikit-learn`
  * `joblib`
    

## 📂 Repository Structure

```bash
customer_churn/
├── artifacts/                # Saved models & pipeline
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── pipeline.pkl
│   └── inference_example.py
├── reports/                  # EDA plots & metrics
│   ├── summary.json
│   ├── cv_baselines.json
│   ├── val_metrics.json
│   ├── test_metrics.json
│   ├── classification_report_test.csv
│   ├── correlation_numeric.png
│   └── ...
├── data/                     # Raw dataset
│   └── telecom_churn.csv
├── churn_pipeline.ipynb      # Main end-to-end notebook
├── churn_pipeline.py         # (Optional) Script version of pipeline
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
└── docs/
    └── PIPELINE.md           # Step-by-step explanation
```


## ⚡ Getting Started

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/telecom-churn.git
cd telecom-churn
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the pipeline

```bash
python churn_pipeline.py
```

By default, the script auto-detects the first `.csv` file in your directory. If needed, update `CSV_PATH` inside `churn_pipeline.py`.


## 📊 Results

* Baseline Models: Logistic Regression, Random Forest, HistGradientBoosting
* Metrics: **ROC-AUC**, **F1-score**, **Accuracy**
* Outputs saved in the `reports/` folder.


## 🔮 Inference Example

```python
import pandas as pd, joblib
pipe = joblib.load("artifacts/pipeline.pkl")
new_data = pd.read_csv("new_customers.csv")  # without target column
proba = pipe.predict_proba(new_data)[:, 1]
pd.DataFrame({"proba_churn": proba}).to_csv("predictions.csv", index=False)
```


## 📌 Next Steps

* Try advanced models: **XGBoost, LightGBM, CatBoost**
* Deploy the pipeline with **FastAPI** or **Streamlit**
* Monitor **data drift** in production


## 🙌 Acknowledgements

* Dataset: [Telecom Churn Dataset on Kaggle](https://www.kaggle.com/datasets/barun2104/telecom-churn)
* Author: Kaggle community dataset by **barun2104**


## ⭐ Support

If you like this project, don’t forget to **star ⭐ the repo**!
