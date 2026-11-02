# 📡 Telecom Customer Churn Prediction | End-to-End Machine Learning

### Predicting Customer Churn and Supporting Data-Driven Retention Strategies

**Python | Pandas | Scikit-learn | LightGBM | Data Analysis | Machine Learning**

## 📌 Project Overview

Customer churn is one of the most important challenges in the telecommunications industry. Understanding why customers cancel their subscriptions allows companies to improve customer retention and make more informed business decisions.

This project was developed as the **Final Data Science Project (Sprint 17) at TripleTen**.

The objective was to build a machine learning solution for **Interconnect**, a fictional telecommunications company, to predict which customers are more likely to cancel their services.

The project covers the complete analytical workflow, from data exploration and preprocessing to model development, performance evaluation, customer risk segmentation, and business recommendations.

## 🎯 Business Objectives

- Analyze customer behavior and service usage.
- Identify patterns associated with customer churn.
- Integrate and preprocess multiple datasets.
- Develop and compare machine learning classification models.
- Optimize predictive performance using cross-validation.
- Identify customers at higher risk of cancellation.
- Translate model predictions into actionable customer retention strategies.

## 🛠️ Technologies Used

| Technology | Application |
|---|---|
| Python | Data analysis and programming |
| Pandas | Data cleaning, transformation, and integration |
| NumPy | Numerical operations |
| Matplotlib & Seaborn | Data visualization |
| Scikit-learn | Machine learning, preprocessing, and evaluation |
| LightGBM | Gradient boosting classification |
| Jupyter Notebook | Development and documentation |

## 📂 Project Structure

```text
telecom-customer-churn-prediction/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── 01_interconnect_data_exploration.ipynb
│   └── 02_interconnect_churn_modeling.ipynb
│
└── data/
    ├── contract.csv
    ├── personal.csv
    ├── internet.csv
    └── phone.csv
```

The `data/` directory represents the original project inputs and should only be published when redistribution is permitted.

## 📊 Dataset Description

The project uses four datasets containing customer information.

| Dataset | Description |
|---|---|
| `contract.csv` | Contract dates, billing information, payment methods, and subscription types |
| `personal.csv` | Customer demographic information |
| `internet.csv` | Internet services and additional subscriptions |
| `phone.csv` | Telephone service information |

The datasets were combined using the customer identifier to create a unified analytical dataset.

### Target Variable

The prediction target is `Churn`, derived from the contract termination information.

- `0` — Customer remains active
- `1` — Customer has canceled the service

The dataset contains **7,043 customers**, including **1,869 churned customers**, representing approximately **26.5%** of the total.

## 🔍 Phase 1 — Exploratory Data Analysis

**Notebook:** `01_interconnect_data_exploration.ipynb`

The first phase focuses on understanding the data, identifying quality issues, and investigating factors associated with customer cancellations.

### Data Preparation

The main steps included:

- Inspecting data types, missing values, and duplicates.
- Converting billing information into numerical formats.
- Creating the churn target variable.
- Calculating customer tenure.
- Merging customer, contract, internet, and phone datasets.
- Handling missing values after integration.
- Identifying variables that could introduce data leakage.

### Key Exploratory Findings

The exploratory analysis revealed several patterns associated with customer churn:

**Contract Type**

Customers with month-to-month contracts showed substantially higher churn rates than customers with annual or two-year agreements.

**Payment Method**

Electronic check payments were associated with higher churn rates compared with automatic payment methods.

**Internet Services**

Customers using fiber optic internet showed higher churn rates than those using DSL.

**Customer Tenure**

Customers with shorter relationships with the company were more likely to cancel their services.

These findings helped guide the machine learning development process.

## 🤖 Phase 2 — Machine Learning Development

**Notebook:** `02_interconnect_churn_modeling.ipynb`

The second phase focuses on building and evaluating predictive classification models.

### Model Development

The following algorithms were evaluated:

1. DummyClassifier — Baseline model
2. Logistic Regression
3. Random Forest
4. LightGBM

The development process included:

- Feature selection and preprocessing.
- Encoding categorical variables.
- Training and test separation.
- Cross-validation.
- Class imbalance analysis.
- Hyperparameter optimization.
- Classification threshold adjustment.
- Final model evaluation.

### Initial Model Comparison

| Model | Cross-Validation ROC-AUC |
|---|---:|
| DummyClassifier | 0.5000 |
| Logistic Regression | 0.8385 |
| Random Forest | 0.8166 |
| LightGBM | 0.8352 |

Following hyperparameter optimization, LightGBM achieved the strongest validation performance.

## 🏆 Final Model — Optimized LightGBM

The selected model used the following hyperparameters:

| Parameter | Value |
|---|---|
| `learning_rate` | 0.05 |
| `n_estimators` | 100 |
| `num_leaves` | 15 |
| `class_weight` | None |
| Classification threshold | 0.30 |

### Final Test Results

| Metric | Result |
|---|---:|
| **ROC-AUC** | **0.8396** |
| Accuracy | 75.59% |
| Precision | 52.81% |
| Recall | 75.40% |
| F1-score | 0.6211 |

The optimized model achieved an ROC-AUC of approximately **0.84** on unseen test data.

By adjusting the classification threshold to 0.30, the model prioritized identifying customers at risk of cancellation.

Of the **374 customers who actually churned in the test set, 282 were correctly identified**, corresponding to a recall of approximately 75%.

## 🎯 Customer Risk Segmentation

The model's predicted probabilities were used to categorize customers into three risk groups.

| Risk Group | Customers | Actual Churn Rate |
|---|---:|---:|
| Low Risk | 710 | 7.46% |
| Medium Risk | 165 | 23.64% |
| High Risk | 534 | 52.81% |

The observed churn rate increased consistently across the three risk groups.

This segmentation provides a practical way to prioritize customer retention initiatives.

### High-Risk Customer Profile

The high-risk segment showed a strong concentration of:

- Month-to-month contracts: 94.57%
- Electronic check payments: 62.92%
- Fiber optic internet subscriptions: 74.16%

The average monthly charge in the high-risk segment was 75.59, compared with 55.05 in the low-risk segment.

## 🔎 Feature Importance

The LightGBM model identified several influential features, including:

- Total charges
- Monthly charges
- Contract type
- Payment method
- Paperless billing
- Multiple lines
- Internet service type
- Online security

These features contributed to the model's predictions, but their importance should not be interpreted as proof of causation.

## 💡 Business Recommendations

Based on the exploratory analysis and predictive results, the following strategies are proposed:

**1. Prioritize High-Risk Customers**

Focus retention efforts on customers with the highest predicted probability of cancellation.

**2. Encourage Long-Term Contracts**

Explore incentives for customers using month-to-month plans to adopt longer subscription agreements.

**3. Review Payment Experiences**

Investigate potential friction associated with electronic check payments.

**4. Improve Service Support**

Evaluate targeted technical support and online security offers for vulnerable customer segments.

**5. Monitor Customer Retention**

Use model predictions to support periodic risk assessments and measure the effectiveness of retention campaigns.

These recommendations are proposed business actions; their effectiveness would need to be validated through operational testing.

## ⚠️ Project Limitations

- The target distribution is imbalanced.
- Lowering the classification threshold increases recall but also produces more false positives.
- The model was evaluated on historical data and may require monitoring before real-world deployment.
- Feature importance does not establish causal relationships.
- The financial impact of retention campaigns was not directly measured.

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Andressa202/telecom-customer-churn-prediction.git
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm jupyter
```

### 3. Open Jupyter Notebook

```bash
jupyter notebook
```

### 4. Execute the notebooks in order

**First:** `01_interconnect_data_exploration.ipynb`

**Second:** `02_interconnect_churn_modeling.ipynb`

Update the dataset paths as needed. The modeling notebook was separated from a larger notebook and may require variables or prepared data from the first phase to be loaded again when executed independently.

## 📈 Conclusion

This project demonstrates an end-to-end machine learning workflow applied to a real-world-style telecommunications business challenge.

The optimized LightGBM model achieved **0.8396 ROC-AUC** and **75.40% recall** on the test dataset, successfully identifying approximately three out of four customers who canceled their services.

In addition to predictive modeling, customer risk segmentation provided valuable insights for prioritizing retention initiatives.

The project showcases practical experience in exploratory data analysis, data preprocessing, classification, hyperparameter optimization, model evaluation, and business-oriented analytical reasoning.

## 👩‍💻 Author

**Andressa Miranda**

Computer Science | Data Analytics | Python | Machine Learning

[GitHub Profile](https://github.com/Andressa202)

*Final Data Science Project — TripleTen, Sprint 17.*
