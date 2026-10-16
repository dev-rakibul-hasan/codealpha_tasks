# CodeAlpha Data Science Internship - Completed Tasks Portfolio

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Status](https://img.shields.io/badge/Internship_Tasks-100%25_Completed-success.svg)](#)

This repository contains the complete, production-ready project implementations for all 4 tasks of the **CodeAlpha Data Science Internship**. Each task features rigorous exploratory data analysis (EDA), statistical preprocessing, multi-model benchmarking, hyperparameter tuning, serialized model artifacts, and interactive inference functions.

**Author:** [Dev Rakibul Hasan](https://github.com/dev-rakibul-hasan)  
**Organization:** [CodeAlpha](https://www.codealpha.tech/)  
**Domain:** Data Science & Machine Learning  

---

## Executive Summary & Task Dashboard

| Task # | Project Name | Domain / Problem Type | Champion Model | Key Performance Metric | Deliverables |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **Task 1** | **Iris Flower Classification** | Multiclass Classification | K-Nearest Neighbors ($k=5$) & SVC | **100% Test Accuracy**, F1: 1.00 | [Notebook](task1.ipynb) \| [Report](task1.md) \| [Model](models/iris_classifier.joblib) |
| **Task 2** | **Unemployment Analysis with Python** | Econometric & Temporal EDA | Statistical & Geospatial Analysis | **COVID-19 Shock:** 9.2% $\rightarrow$ 20.3% | [Notebook](unemployment_analysis/unemployment_analysis.ipynb) \| [Report](unemployment_analysis/unemployment_analysis.md) |
| **Task 3** | **Car Price Prediction** | Supervised Regression | Log-Transformed Gradient Boosting | **$R^2 = 0.8462$**, MAE: 0.925 Lakhs | [Notebook](car_price_prediction/car_price_prediction.ipynb) \| [Report](car_price_prediction/car_price_prediction.md) \| [Model](models/car_price_predictor.joblib) |
| **Task 4** | **Sales Prediction using Python** | Marketing Mix Modeling & Regression | Gradient Boosting (with Synergy) | **$R^2 = 0.9831$**, MAE: 0.618 units | [Notebook](sales_prediction/sales_prediction.ipynb) \| [Report](sales_prediction/sales_prediction.md) \| [Model](models/sales_prediction_model.joblib) |

---

## Table of Contents
1. [Task 1: Iris Flower Classification](#task-1-iris-flower-classification)
2. [Task 2: Unemployment Analysis with Python](#task-2-unemployment-analysis-with-python)
3. [Task 3: Car Price Prediction with Machine Learning](#task-3-car-price-prediction-with-machine-learning)
4. [Task 4: Sales Prediction using Python](#task-4-sales-prediction-using-python)
5. [Repository Structure](#repository-structure)
6. [Installation & Getting Started](#installation--getting-started)
7. [Dependencies & Technologies](#dependencies--technologies)
8. [License & Acknowledgments](#license--acknowledgments)

---

## Task 1: Iris Flower Classification

* **Directory:** Root / [`iris_flower_classification/`](iris_flower_classification/)
* **Primary Notebook:** [`task1.ipynb`](task1.ipynb)
* **Markdown Report:** [`task1.md`](task1.md)
* **Dataset:** [`Iris.csv`](Iris.csv) (150 samples, 4 biometric features, 3 balanced species)
* **Exported Artifact:** [`models/iris_classifier.joblib`](models/iris_classifier.joblib)

### Key Methodologies
* **Exploratory Data Analysis:** Analyzed feature histograms with KDE, class balance, pairplots, and correlation heatmaps. Found that *Iris-setosa* is linearly separable via `PetalLengthCm` and `PetalWidthCm`.
* **Preprocessing:** Handled ID column removal, stratified 80:20 train-test split, and standardized numerical predictors via `StandardScaler`.
* **Benchmarking:** Evaluated Logistic Regression, K-Nearest Neighbors, Support Vector Machine (RBF kernel), Decision Trees, and Random Forests using 5-fold Stratified Cross-Validation.
* **Hyperparameter Tuning:** Conducted an exhaustive search for optimal $k$ in KNN ($k \in [1, 20]$), finding $k=5$ yields the highest cross-validated generalization score.

### Benchmark Results
| Model | CV Mean Accuracy (%) | CV Std Dev (%) | Test Accuracy (%) |
| :--- | :---: | :---: | :---: |
| **K-Nearest Neighbors ($k=5$)** | **96.67%** | $\pm 3.33\%$ | **100.00%** |
| **Support Vector Machine (RBF)** | **96.67%** | $\pm 3.33\%$ | **100.00%** |
| **Logistic Regression** | **95.83%** | $\pm 3.12\%$ | **100.00%** |
| **Random Forest Classifier** | **95.00%** | $\pm 3.12\%$ | **96.67%** |
| **Decision Tree Classifier** | **92.50%** | $\pm 5.00\%$ | **96.67%** |

---

## Task 2: Unemployment Analysis with Python

* **Directory:** [`unemployment_analysis/`](unemployment_analysis/)
* **Primary Notebook:** [`unemployment_analysis/unemployment_analysis.ipynb`](unemployment_analysis/unemployment_analysis.ipynb)
* **Markdown Report:** [`unemployment_analysis/unemployment_analysis.md`](unemployment_analysis/unemployment_analysis.md)
* **Dataset:** [`unemployment_analysis/unemployment.csv`](unemployment_analysis/unemployment.csv)

### Key Methodologies
* **Data Hygiene & Engineering:** Resolved irregular whitespace formatting across column headers, mapped geographic coordinate points (`longitude`, `latitude`), and classified the temporal observation window into 3 epidemiological phases:
  1. *Pre-Lockdown* (January - March 2020)
  2. *Lockdown Peak* (April - June 2020)
  3. *Unlock / Recovery* (July - October 2020)
* **Macro Timeline & Shock Analysis:** Traced national monthly mean unemployment trajectories alongside Labour Participation Rates (LPR), revealing a dramatic spike following the nationwide lockdown on March 24, 2020.
* **Geographical & Zonal Disparities:** Benchmarked economic volatility across the 5 macro-zones (North, South, East, West, Northeast) and ranked the top 10 most impacted vs resilient states.

### Key Insights & Findings
* **The Lockdown Shock:** National average unemployment more than doubled from **~9.22%** in the pre-lockdown phase to an alarming peak of **20.30%** during April-June 2020.
* **V-Shaped Rebound:** As economic restrictions relaxed from July onwards, the unemployment rate rapidly recovered to **10.14%**, indicating resilient latent labor demand.
* **State Disparities:** States with heavy industrial or migrant worker exposure suffered extreme disruptions (Puducherry: 58.0%, Jharkhand: 44.9%, Bihar: 39.7%), whereas agrarian states (Meghalaya, Assam, Odisha) displayed significantly higher structural resilience.
* **Policy Takeaway:** Recommended creating an urban counterpart to MGNREGA to protect informal urban wage-earners during emergency closures.

---

## Task 3: Car Price Prediction with Machine Learning

* **Directory:** [`car_price_prediction/`](car_price_prediction/)
* **Primary Notebook:** [`car_price_prediction/car_price_prediction.ipynb`](car_price_prediction/car_price_prediction.ipynb)
* **Markdown Report:** [`car_price_prediction/car_price_prediction.md`](car_price_prediction/car_price_prediction.md)
* **Dataset:** [`car_price_prediction/car data.csv`](car_price_prediction/car%20data.csv)
* **Exported Artifact:** [`models/car_price_predictor.joblib`](models/car_price_predictor.joblib)

### Key Methodologies
* **Feature Engineering:** Calculated dynamic vehicle age (`Car_Age = 2024 - Year`), handled categorical feature cardinality (`Fuel_Type`, `Selling_type`, `Transmission`), and resolved duplicate records.
* **Target Transformation:** Applied `TransformedTargetRegressor` with a logarithmic transformation ($\log(1+y)$) to stabilize right-skewed car pricing variance and eliminate heteroscedasticity.
* **Pipeline Architecture:** Integrated `OneHotEncoder(handle_unknown='ignore')` and `StandardScaler` inside a scikit-learn `Pipeline`.
* **Model Benchmark:** Benchmarked Linear Regression, Ridge, Decision Trees, Random Forest Regressors, and Gradient Boosting Regressors.

### Benchmark Results
| Model | Cross-Validation $R^2$ | Test $R^2$ Score | Test MAE (Lakhs INR) | Test RMSE (Lakhs INR) |
| :--- | :---: | :---: | :---: | :---: |
| **Log-Transformed Gradient Boosting (Champion)** | **0.8961** | **0.8462** | **0.9258** | **2.0012** |
| **Standard Gradient Boosting Regressor** | 0.8961 | 0.7480 | 1.1523 | 2.5604 |
| **Ridge Regression** | 0.8854 | 0.7529 | 1.4805 | 2.5350 |
| **Linear Regression** | 0.8846 | 0.7527 | 1.4725 | 2.5361 |
| **Random Forest Regressor** | 0.8956 | 0.4614 | 1.5250 | 3.7547 |

* **Feature Importance:** `Present_Price` accounted for over **68%** of the decision-tree split gain, followed by `Car_Age` (**18%**) and `Driven_kms` (**8%**).

---

## Task 4: Sales Prediction using Python

* **Directory:** [`sales_prediction/`](sales_prediction/)
* **Primary Notebook:** [`sales_prediction/sales_prediction.ipynb`](sales_prediction/sales_prediction.ipynb)
* **Markdown Report:** [`sales_prediction/sales_prediction.md`](sales_prediction/sales_prediction.md)
* **Dataset:** [`sales_prediction/advertising.csv`](sales_prediction/advertising.csv)
* **Exported Artifact:** [`models/sales_prediction_model.joblib`](models/sales_prediction_model.joblib)

### Key Methodologies
* **Marketing Mix Modeling:** Analyzed individual and cross-channel elasticity for TV, Radio, and Newspaper advertising spends on product sales volume.
* **Synergy Engineering:** Formulated a cross-media interaction predictor (`TV_Radio_Synergy = TV * Radio`) to model the compounding interaction between mass television awareness and targeted radio repetition.
* **Econometric Attribution:** Calculated marginal return on ad spend (ROAS) coefficients via Ordinary Least Squares (OLS) regression.
* **Machine Learning Pipeline:** Built an ensemble pipeline with `StandardScaler` and `GradientBoostingRegressor`.

### Benchmark Results
| Model | CV Mean $R^2$ | Test $R^2$ Score | Test MAE | Test RMSE |
| :--- | :---: | :---: | :---: | :---: |
| **Gradient Boosting Regressor (Champion)** | **0.9672** | **0.9831** | **0.6187** | **0.8475** |
| **Random Forest Regressor** | **0.9663** | **0.9813** | **0.6203** | **0.8912** |
| **Multiple Linear Regression (with Synergy)** | 0.9650 | 0.9742 | 0.6718 | 1.0480 |
| **Standard Multiple Linear Regression** | 0.8703 | 0.8994 | 1.4608 | 1.7816 |
| **Ridge Regression** | 0.8703 | 0.8994 | 1.4608 | 1.7816 |

### Econometric Attribution & Budget Advice
* **Radio:** Highest marginal lift at **+0.188 units** per \$1,000 spend.
* **TV:** Strong, steady return at **+0.045 units** per \$1,000 spend.
* **Newspaper:** Statistically insignificant return (**+0.001 units** per \$1,000 spend).
* **Strategic Recommendation:** Reallocate newspaper budget into synchronized TV and Radio campaigns to unlock the cross-channel synergy multiplier.

---

## Repository Structure

```plaintext
codealpha_tasks/
├── data/                               # Master datasets directory
│   ├── Iris.csv
│   ├── unemployment.csv
│   ├── car data.csv
│   └── advertising.csv
├── models/                             # Serialized trained model bundles
│   ├── iris_classifier.joblib
│   ├── car_price_predictor.joblib
│   └── sales_prediction_model.joblib
├── iris_flower_classification/         # Task 1 directory
│   ├── task1.ipynb
│   ├── task1.md
│   ├── Iris.csv
│   └── task1_files/
├── unemployment_analysis/              # Task 2 directory
│   ├── unemployment_analysis.ipynb
│   ├── task2.ipynb
│   ├── unemployment_analysis.md
│   ├── task2.md
│   ├── unemployment.csv
│   └── unemployment_analysis_files/
├── car_price_prediction/               # Task 3 directory
│   ├── car_price_prediction.ipynb
│   ├── task3.ipynb
│   ├── task2.ipynb
│   ├── car_price_prediction.md
│   ├── task3.md
│   ├── task2.md
│   ├── car data.csv
│   ├── car_price_predictor.joblib
│   └── car_price_prediction_files/
├── sales_prediction/                   # Task 4 directory
│   ├── sales_prediction.ipynb
│   ├── task4.ipynb
│   ├── sales_prediction.md
│   ├── task4.md
│   ├── advertising.csv
│   ├── sales_prediction_model.joblib
│   └── sales_prediction_files/
├── scripts/                            # Notebook build & automation scripts
│   ├── build_task1.py
│   ├── build_task2.py
│   ├── build_task3.py
│   └── build_task4.py
├── task1.ipynb                         # Task 1 root convenience link
├── task1.md
├── Iris.csv
├── requirements.txt                    # Project dependencies
├── .gitignore
└── README.md                           # Master documentation
```

---

## Installation & Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/dev-rakibul-hasan/codealpha_tasks.git
cd codealpha_tasks
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python3 -m venv .venv

# Activate environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run Any Jupyter Notebook
```bash
jupyter notebook
```
Open any of the task notebooks (`task1.ipynb`, `unemployment_analysis/unemployment_analysis.ipynb`, `car_price_prediction/car_price_prediction.ipynb`, or `sales_prediction/sales_prediction.ipynb`) to view the interactive plots and run inference.

---

## Dependencies & Technologies

* **Programming Language:** Python 3.10+
* **Data Processing & Manipulation:** Pandas, NumPy
* **Machine Learning & Modeling:** Scikit-Learn
* **Model Serialization:** Joblib
* **Data Visualization:** Matplotlib, Seaborn
* **Interactive Development:** Jupyter Notebook, IPyKernel, NbConvert

---

## Author & Acknowledgments

* **Developer:** [Dev Rakibul Hasan](https://github.com/dev-rakibul-hasan)
* **Organization:** [CodeAlpha](https://www.codealpha.tech/)
* **Internship Program:** CodeAlpha Data Science Internship Program

*Completed with pride as part of the CodeAlpha Data Science track.*
