# 🧭 Customer Churn Pipeline - Step-by-Step Explanation

Let me explain **what each step did and why**.


## 🔎 Step-by-Step Explanation

### Step 1 - Load the dataset

* **What we did:**

  * Read the CSV file with `pandas.read_csv`.
  * Printed shape, columns, and first few rows.
* **Why:**

  * To make sure we have the right dataset loaded.
  * It’s always the first sanity check: correct path, correct columns, correct size.

### Step 2 - Exploratory Data Analysis (EDA)

* **What we did:**

  * Checked column data types with `.info()`.
  * Looked at summary statistics (`.describe()`).
  * Counted missing values.
  * Plotted:

    * Target distribution (class balance).
    * Missing value counts.
    * Correlation heatmap for numeric features.
    * Histograms for top numeric features.
* **Why:**

  * EDA helps us **understand the data’s structure, scale, and quality**.
  * Plots highlight class imbalance (important for churn tasks).
  * Correlations show redundancy between features.
  * Histograms reveal skewness or outliers.

### Step 3 - Data cleaning

* **What we did:**

  * Dropped useless ID-like columns (`customerID`).
  * Removed duplicates.
  * Coerced numeric-looking object columns (e.g., `"42"` → `42.0`).
  * Dropped rows with missing target values.
* **Why:**

  * IDs don’t help prediction (they’re unique identifiers, not features).
  * Duplicates distort distributions.
  * Stringified numerics cause problems with models.
  * Missing target rows can’t be used in supervised learning.

### Step 4 - Split into Train, Validation, Test sets

* **What we did:**

  * Split dataset:

    * 70% → Train
    * 15% → Validation
    * 15% → Test
  * Stratified splits (preserve class balance).
  * Identified “positive class” (in churn, usually `1` = churn).
* **Why:**

  * Prevents data leakage: test set is held back until final evaluation.
  * Validation set is used to pick thresholds and tune hyperparameters.
  * Train set is used for model fitting.
  * Stratification ensures churn vs. non-churn balance across splits.

### Step 5 - Preprocessing pipeline

* **What we did:**

  * Created numeric pipeline: median imputation + standard scaling.
  * Created categorical pipeline: mode imputation + one-hot encoding.
  * Combined them into a `ColumnTransformer`.
* **Why:**

  * Real-world data is messy: we need to **handle missing values**.
  * Models often expect standardized features.
  * One-hot encoding lets models handle categorical variables.
  * Putting this in a pipeline ensures transformations apply consistently.

### Step 6 - Baseline models with cross-validation

* **What we did:**

  * Trained 3 models: Logistic Regression, Random Forest, HistGradientBoosting.
  * Used 5-fold stratified CV on the train set.
  * Evaluated metrics: ROC-AUC, F1, Accuracy.
  * Picked the best baseline model.
* **Why:**

  * Always start with multiple baselines.
  * CV gives a stable estimate, not just one split’s performance.
  * Different models capture different patterns:

    * Logistic Regression → linear baseline.
    * Random Forest → bagged trees (robust).
    * HistGB → gradient boosting (often best for tabular).

### Step 7 - Hyperparameter tuning

* **What we did:**

  * Performed `RandomizedSearchCV` with predefined parameter grids.
  * Tuned only the best baseline model.
  * Used ROC-AUC as the objective.
* **Why:**

  * Default parameters aren’t optimal.
  * Hyperparameter tuning finds the best bias/variance trade-off.
  * Randomized search is faster than grid search while covering a wide space.

### Step 8 - Validation set evaluation & threshold tuning

* **What we did:**

  * Got churn probabilities on the validation set.
  * Found the threshold that maximized F1 score.
  * Calculated ROC-AUC, Accuracy, F1 at that threshold.
* **Why:**

  * Models output probabilities, not just 0/1.
  * Default 0.5 cutoff isn’t always optimal (especially with class imbalance).
  * Tuning the threshold improves practical performance (e.g., catching more churners).

### Step 9 - Final evaluation on Test set

* **What we did:**

  * Applied the tuned model and threshold on the held-out test set.
  * Measured ROC-AUC, Accuracy, F1.
  * Saved plots: ROC curve, confusion matrix.
  * Saved classification report.
* **Why:**

  * The test set gives an **unbiased estimate** of how the model performs on unseen data.
  * Plots help visualize performance trade-offs.
  * This step tells us if our model generalizes well.

### Step 10 - Feature importance (Permutation Importance)

* **What we did:**

  * Refit the tuned pipeline on Train+Validation.
  * Computed permutation importance on the Test set.
  * Saved CSV + bar chart of top 25 features.
* **Why:**

  * Permutation importance shows how much each feature contributes to predictive power.
  * This helps interpret the model (e.g., is churn driven by `CustServCalls` or `ContractRenewal`?).
  * Feature importance guides future feature engineering.

### Step 11 - Save artifacts + inference helper

* **What we did:**

  * Saved:

    * `preprocessor.pkl` → transformations only
    * `model.pkl` → tuned model only
    * `pipeline.pkl` → full pipeline (preprocessing + model)
  * Saved `metadata.json` with params, metrics, etc.
  * Wrote an `inference_example.py` script for future predictions.
* **Why:**

  * Artifacts let you reuse the trained model without retraining.
  * Having both model + preprocessing ensures consistency at inference time.
  * Metadata captures what was run (reproducibility).
  * The helper script makes deployment or batch scoring easy.


✅ In summary:

* **Steps 1–3:** Get the dataset ready.
* **Steps 4–5:** Make it ML-friendly.
* **Steps 6–7:** Try models and tune them.
* **Steps 8–9:** Validate and test the model fairly.
* **Step 10:** Understand the model.
* **Step 11:** Save everything for reuse/deployment.
