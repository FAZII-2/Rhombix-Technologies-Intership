# Credit Scoring Model

An end-to-end credit risk classification system built on the "Give Me Some Credit" dataset (150,000 applicants), covering rigorous data inspection, advanced EDA, a leakage-safe `scikit-learn` pipeline, a five-model linear classifier comparison (Logistic Regression, Ridge, Lasso, LDA, Linear SVM), hyperparameter tuning, model interpretability, an interactive prediction demo, and a full CI/CD/CT (Continuous Integration / Deployment / Training) setup.

Built as part of the Machine Learning Internship at Rhombix Technologies.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Data Inspection Findings](#data-inspection-findings)
- [Data Cleaning](#data-cleaning)
- [Advanced EDA](#advanced-eda)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Models Trained](#models-trained)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Final Model Comparison](#final-model-comparison)
- [Final Model Selection & Interpretability](#final-model-selection--interpretability)
- [Interactive Prediction Demo](#interactive-prediction-demo)
- [Limitations & Honest Tradeoffs](#limitations--honest-tradeoffs)
- [How to Run](#how-to-run)
- [Repository Structure](#repository-structure)
- [Author](#author)

---

## Project Overview

**Goal:** Build a credit scoring model that predicts applicant creditworthiness using features like income, debt, and credit history — framed as a binary classification problem (will this applicant become seriously delinquent within 2 years, or not).

**What this project demonstrates:**
- A genuine data-inspection-first workflow — investigating data quality issues with evidence before applying any fix, rather than cleaning on assumption
- A leakage-safe, reusable `scikit-learn` `Pipeline` with a custom-built transformer
- A comparison of five linear classification approaches, including the mathematical reasoning behind each
- Real hyperparameter tuning via cross-validation, including an honest report of when regularization *didn't* help and why
- Model interpretability suited to a regulated, explainability-sensitive domain like credit scoring

---

## Dataset

**Source:** ["Give Me Some Credit"](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle)

- **Size:** 150,000 applicant records
- **Target:** `SeriousDlqin2yrs` — whether the applicant experienced 90+ days past-due delinquency within 2 years (1 = Yes, 0 = No)
- **Class balance:** Imbalanced — 93.32% Good Standing / 6.68% Became Delinquent
- **Features:** 10 financial/demographic variables covering income, debt ratio, credit utilization, credit line history, late-payment counts, real estate loans, age, and dependents

**Why this dataset:** its features map directly to the project brief (income, debt, credit history), and its realistic messiness — missing values, coded sentinel values, extreme outliers — gave genuine material for a proper data inspection phase, rather than a pre-cleaned dataset with nothing left to investigate.

---

## Project Workflow

1. Discussion & planning (model family, dataset sourcing, target framing)
2. Raw data inspection (structure, types, missing values, descriptive stats, target balance)
3. Investigation of specific data-quality issues, with evidence, before fixing anything
4. Data cleaning (encoded as a reusable function, then a custom pipeline transformer)
5. Advanced exploratory data analysis
6. Preprocessing pipeline construction (`ColumnTransformer`-style cleaning + scaling)
7. Training five linear models
8. Hyperparameter tuning (cross-validation)
9. Final model selection & interpretability
10. Interactive prediction demo
11. Documentation

---

## Data Inspection Findings

Before any cleaning, the raw dataset was inspected directly, surfacing five real issues:

| # | Issue | Evidence |
|---|---|---|
| 1 | Missing `MonthlyIncome` | 19.82% of rows (29,731) |
| 2 | Missing `NumberOfDependents` | 2.62% of rows (3,924) |
| 3 | Sentinel codes (98/96) in the three late-payment columns | 269 rows (0.18%), identical across all 3 columns — delinquency rate within these rows was 54.65% vs. 6.68% overall |
| 4 | `age` = 0 | A single row, isolated data-entry error |
| 5 | Extreme outliers in `RevolvingUtilizationOfUnsecuredLines` and `DebtRatio` | 99th percentile of `DebtRatio` alone was already ~4,979; found to be driven almost entirely (92%) by rows with missing income |

Each issue was investigated with real evidence (value counts, cross-tabulations against the target, percentile breakdowns) before deciding on a fix — rather than applying a generic cleaning routine on assumption.

---

## Data Cleaning

Fixes applied, each designed to preserve signal rather than silently discard it:

- **Missing income** → binary `Income_Was_Missing` flag + median imputation
- **Missing dependents** → binary `Dependents_Was_Missing` flag + imputed with 0
- **Sentinel codes** → binary `Has_Sentinel_Code` flag (captures the strong 54.65% delinquency signal) + replaced with the column's non-sentinel median
- **Age = 0** → median imputation
- **`RevolvingUtilizationOfUnsecuredLines` outliers** → capped at the 99th percentile
- **`DebtRatio` skew** → log-transformed (`log1p`) rather than capped alone, since capping left the bulk of the distribution still heavily right-skewed — a genuine problem for linear models, which assume well-scaled, roughly linear feature contributions

**Multicollinearity finding:** `Income_Was_Missing` and the log-transformed `DebtRatio` showed a 0.87 correlation (76% shared variance) — both features were kept, with the tradeoff and its effect on model coefficients documented directly (see Hyperparameter Tuning below).

---

## Advanced EDA

Beyond basic distributions, EDA specifically examined:
- Age and income distributions split by outcome
- Late-payment history, debt ratio, and credit utilization by outcome (box/violin plots)
- Delinquency **rate** trends across real estate loan counts (a monotonic increase observed)
- A full feature correlation heatmap, in plain-language labels
- A dedicated investigation of the `Income_Was_Missing` / `DebtRatio` multicollinearity, visualized directly

---

## Preprocessing Pipeline

A custom `CreditDataCleaner` class (inheriting `BaseEstimator`, `TransformerMixin`) was built to encode all five cleaning fixes as a proper `scikit-learn` transformer:

- **`.fit()`** learns all statistics (medians, the outlier cap) **only from training data**
- **`.transform()`** applies those learned values to any dataset, without recalculating anything from it

This was combined with `StandardScaler` into a single `Pipeline` object — meaning cleaning, scaling, and modeling are one deployable unit, safely reusable on new applicant data without any manual reimplementation of the cleaning logic.

---

## Models Trained

All five models were evaluated as a linear classifier comparison, per the project's brief:

| Model | Core Idea |
|---|---|
| **Logistic Regression** | Models P(delinquent) directly via the sigmoid function on a weighted feature sum; trained by minimizing log loss |
| **Ridge Logistic Regression (L2)** | Adds a squared-coefficient penalty (`λ·Σwᵢ²`) to shrink coefficients smoothly, addressing multicollinearity |
| **Lasso Logistic Regression (L1)** | Adds an absolute-value penalty (`λ·Σ|wᵢ|`), capable of shrinking weak coefficients to exactly zero (automatic feature selection) |
| **Linear Discriminant Analysis (LDA)** | Models each class as a Gaussian distribution with shared covariance, and derives the linear boundary from class means |
| **Linear SVM** | Finds the maximum-margin linear boundary; requires extra probability calibration (`CalibratedClassifierCV`), since it has no native `predict_proba` |

All models used `class_weight='balanced'` where supported, given the dataset's 93.3%/6.7% class imbalance — LDA does not support this option, which limits its comparability on recall.

---

## Hyperparameter Tuning

Ridge and Lasso were tuned via 3-fold cross-validation across `C ∈ [0.001, 0.01, 0.1, 1, 10, 100]`, optimizing for ROC-AUC.

**Result:** both settled on **C = 10** (a *weak* regularization strength) — confirming that with 120,000 training rows against only 13 features, the dataset simply didn't need strong regularization to produce stable coefficients. This was a validated, evidence-backed conclusion rather than an assumption.

**A genuinely interesting finding along the way:** at an artificially strong penalty (`C = 0.01`), several coefficients — including `NumberOfTimes90DaysLate` — flipped sign entirely, despite near-identical overall accuracy/AUC. This demonstrated that excessive regularization doesn't automatically mean "more correct" — with multiple correlated late-payment features, an over-penalized model can settle on a mathematically valid but counter-intuitive (and far less interpretable) coefficient distribution.

---

## Final Model Comparison

| Model | Accuracy | ROC-AUC | Recall (Delinquent) |
|---|---|---|---|
| **Logistic Regression** | 0.8042 | **0.8609** | **0.754** |
| Ridge (tuned, C=10) | 0.8042 | 0.8609 | 0.754 |
| Lasso (tuned, C=10) | 0.8042 | 0.8609 | 0.754 |
| LDA | 0.9329 | 0.8607 | 0.311 |
| Linear SVM | 0.9373 | 0.8577 | 0.166 |

**Key finding:** accuracy is a misleading metric here. LDA and SVM post higher accuracy purely by leaning toward the majority class — their recall on the class that actually matters (catching genuinely risky applicants) is dramatically worse. ROC-AUC, a threshold-independent metric, shows all five models have similarly strong underlying discriminative ability (0.858–0.861) — the real difference lies in how each handles the class imbalance at decision time.

---

## Final Model Selection & Interpretability

**Selected model: plain (unregularized) Logistic Regression.**

Reasoning: best ROC-AUC (tied at the top), by far the best recall on the minority class, no extra calibration machinery required (unlike SVM), and fully interpretable coefficients — directly satisfying the explainability that real-world credit decisions require.

**Top features increasing predicted risk:** Credit Card Usage Ratio, Times 90+ Days Late, Times 30-59 Days Late, Times 60-89 Days Late
**Top features decreasing predicted risk:** Age, Monthly Income, Debt-to-Income Ratio (log-transformed)

---

## Interactive Prediction Demo

A `predict_applicant(...)` function accepts raw applicant details (income, age, debt ratio, credit utilization, late-payment history, etc.) and returns a predicted delinquency probability plus a visual risk-meter chart. Because the underlying object is the full trained `Pipeline`, raw uncleaned applicant data can be passed directly — no manual preprocessing required.

---

## Limitations & Honest Tradeoffs

- **LDA's Gaussian/equal-covariance assumptions** are almost certainly violated by this data's skewed financial features, and it lacks `class_weight` support — its high accuracy is a known imbalance artifact, not a genuine strength.
- **Linear SVM's probability calibration** (`CalibratedClassifierCV`) adds a layer of approximation not present in the other models' native probability outputs.
- **Regularization (Ridge/Lasso) provided no measurable improvement** over the unregularized baseline in this case — reported transparently, since the dataset's size-to-feature ratio simply didn't require it.
- **The Income_Was_Missing / DebtRatio multicollinearity** was documented but not resolved by removing a feature — both were kept, relying on the (ultimately mild) regularization to manage the overlap.
- **CI/CD/CT infrastructure was built but not deployed live** in this project iteration.

---

## How to Run

1. Open the notebook in **Google Colab**
2. Run all cells top to bottom — a Kaggle API token may be required on first run to download the dataset via `kagglehub`
3. For the prediction demo cell, call `predict_applicant(...)` with your own applicant details

To activate CI/CD/CT: copy the `/credit_scoring` folder (see structure below) into your GitHub repository, including the dataset under `data/cs-training.csv`, then enable GitHub Actions and workflow write permissions in repo Settings.

---

## Repository Structure

```
credit_scoring/
│
├── README.md
├── Credit_Scoring_Model.ipynb        <- Full Colab notebook (all phases)
└──  requirements.txt

```

---

## Author

**Faizan**
Machine Learning Internship — Rhombix Technologies
