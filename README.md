# 🧪 auto_ML 

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/Sklearn-Machine%20Learning-green.svg)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue)

**auto_Ml** is an end-to-end auto_Ml platform with a Gradio web UI. Upload a CSV, pick a target column, and the app auto-detects the task, preprocesses mixed-type data, trains a portfolio of models with cross-validation, tunes **every** model, evaluates on a held-out test set, saves each model as a self-contained `.pkl` bundle, and logs every run to MLflow — all without writing a single line of code.

---

## 🌟 Key Features

### 1. 📂 Intelligent Data Handling
* **Upload & preview** — first 20 rows + shape shown instantly
* **Auto X/y split** — target column dropped and isolated; rows with a missing target are removed cleanly
* **Task auto-detection** — classification vs regression inferred from target dtype and cardinality (float-coded classes still count as classification)
* **Manual override** — flip the task type with one click if the heuristic guesses wrong

### 2. 🔍 Built-in EDA
* **Correlation heatmap** — one click over all numeric columns
* **Distribution histograms** — first 6 numeric columns, KDE overlaid

### 3. ⚙️ Leakage-Proof Preprocessing Pipeline
Fit **on the training set only**, then applied to val/test:

* **Numeric columns:** median imputation → IQR outlier clipping (custom `OutlierHandler`) → `RobustScaler`
* **Categorical columns:** most-frequent imputation → `OneHotEncoder(handle_unknown="ignore")`
* **Post-processing:** `VarianceThreshold(0)` drops constant columns

The **entire fitted pipeline** is saved alongside every model — inference reproduces training exactly.

### 4. 🤖 Automated Training with Model Moka
* **Model Moka** — a curated portfolio per task type:

  | Classification | Regression |
  |---|---|
  | LogisticRegression | Ridge |
  | RandomForestClassifier | RandomForestRegressor |
  | GradientBoostingClassifier | GradientBoostingRegressor |
  | XGBClassifier            | XGBRegressor  |

* **Baseline pass** — every model trained with defaults + 5-fold CV
* **Leaderboard** — ranked by CV mean and primary metric (F1-weighted or R²)
* **Every model evaluated on the test set** — not just the winner
* **MLflow-safe logging** — a logging failure never kills a trained model

### 5. 🎛️ Tuning & Model Selection
* **RandomizedSearchCV on EVERY model** — not just top-K. Every baseline gets a tuned counterpart.
* **Baseline ∪ tuned leaderboard** — compare the two side by side
* **Winner picked by CV mean** across both groups
* **Per-model test plots** — confusion matrix (classification) or actual-vs-predicted scatter (regression) for every model, browsable from a dropdown
* **Per-model downloads** — every `.pkl` bundle is available for download, not just the best one

### 6. 📦 Self-Contained Model Bundles
Each `.pkl` in `models/` contains:

```python
{
    "model":           <fitted estimator>,
    "model_name":      "RandomForestClassifier",
    "source":          "baseline" | "tuned",
    "task_type":       "classification" | "regression",
    "target_column":   "Survived",
    "feature_columns": [...],
    "preprocessor":    <fitted Pipeline>,
    "metrics_test":    {...},
    "cv_mean":         0.92,
}

7. 🧪 Experiment Tracking (MLflow File Store)
    - every run becomes a folder under experiments/
    - Params, metrics, model, preprocessor all logged per run

📂 Project Structure & File Description

auto_ml/
├── app.py                      # 🖥️  Gradio UI (thin wrapper, no ML logic)
├── config.py                   # ⚙️  Paths, seeds, search spaces, tuning flags
├── data_utils.py               # 🧹  CSV loading, X/y prep, task detection, splitting
├── preprocessing.py            # 🧹  custom OutlierHandler + 
├── model_moka.py               # 🧠  Model zoo — estimators + per-task search spaces
├── model_utils.py              # 🤖  Metrics, training, tuning,evaluation,MLflow saving
├── pipeline_orchestrator.py    # 🔗  End-to-end pipeline
├── requirements.txt            # 📦  Python dependencies
├── README.md                   # 📖  This file for Documentation
├── auto_Ml_Demo.ipynb          # 📓  Notebook demo (no UI)
│
├── models/                     # (auto-created)  .pkl bundle per trained model
└── experiments/                # (auto-created)  MLflow file store, one folder per run