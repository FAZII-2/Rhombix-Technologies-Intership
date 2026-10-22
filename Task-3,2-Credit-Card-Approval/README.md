# Credit Card Approval Prediction (Logistic Regression + GridSearchCV)

Rhombix Technologies ML Internship, Task 3.2.
Notebook: `credit_card_approval.ipynb` | Kaggle notebook: [link] | Dataset: [rikdifos/credit-card-approval-prediction](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction)

## Summary
I built a credit-risk classifier from application data using Logistic Regression tuned with GridSearchCV. The workflow covers missing-value handling, categorical encoding, feature scaling, class imbalance, grouped cross-validation and a model comparison.

**Main finding:** the application features carry very little signal about future delinquency. Tuned Logistic Regression reaches a grouped-CV ROC-AUC of about 0.57 and a test ROC-AUC of 0.505 (95% bootstrap CI 0.437 to 0.573, which includes 0.5). Non-linear models and alternative label definitions did not change this. I report the result as it is, without tuning toward the test set.

## Data
- `application_record.csv`: 438,557 applications, 18 columns.
- `credit_record.csv`: monthly payment status (1,048,575 rows, apparently truncated at the spreadsheet row limit, covering 45,985 applicants).
- Only 36,457 applicants appear in both files, so the modeling set has 36,457 rows.
- **Target:** `bad = 1` if the applicant was ever 60+ days overdue (STATUS 2 to 5). Bad rate: 1.69% (616 of 36,457), about 58 good applicants per bad one.

## Method
1. **Raw EDA** (8 figures): target balance, missingness, distributions, bad rate by category and by binned numerics, correlations, leakage check.
2. **Cleaning:**
   - Dropped the constant `FLAG_MOBIL`.
   - Dropped `n_months` and `history_start`, which are leakage: they describe how long an applicant was observed, not information known at application time.
   - Converted day-offsets to years and replaced the pensioner placeholder (365243) with `IS_EMPLOYED` and `EMPLOYED_YEARS = 0`.
   - Filled missing occupation (31%, almost entirely pensioners) with an explicit "Unknown" category.
   - Merged two tiny categories (Student, Academic degree), capped extreme counts, and log-transformed income.
3. **Duplicate profiles:** 91.5% of rows share identical features with another row, giving only 9,691 unique profiles. This is likely one person holding several IDs, which I could not verify. A random split would place twins in both train and test, so I used `StratifiedGroupKFold` grouped by profile for the train/test split and all cross-validation. Zero profiles are shared between train and test.
4. **Processing EDA:** scaling before/after, correlations, and VIF. Collinear features (`CNT_FAM_MEMBERS`, `IS_EMPLOYED`) were dropped, reducing features with VIF above 10 from 4 to 2 (both housing dummies, a coding artifact). Figures 01 to 05 in `eda_processed` predate this fix (04 is the "before" VIF; 06 is the "after").
5. **Imbalance:** compared plain LR, class weights, SMOTE and undersampling under grouped CV, with resampling inside training folds only.
6. **Tuning:** GridSearchCV over C, penalty (L1/L2) and class weight, refit on PR-AUC (36 candidates, 5 grouped folds).
7. **Evaluation:** thresholds chosen on out-of-fold train predictions only, one-shot test evaluation, grouped bootstrap intervals, decile lift, calibration, odds ratios.
8. **Comparison:** Dummy, tuned LR, Random Forest, HistGradientBoosting.
9. **Label sensitivity:** 30+, 60+ and 90+ days overdue.

## Results

| Model | CV ROC-AUC | CV PR-AUC | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|---|
| Dummy | 0.500 | 0.017 | 0.500 | 0.015 |
| **Logistic Regression (tuned)** | 0.565 | 0.030 | 0.505 | 0.026 |
| Random Forest | 0.570 | 0.033 | 0.556 | 0.018 |
| HistGradientBoosting | 0.573 | 0.028 | 0.575 | 0.037 |

Fold-to-fold standard deviation of CV ROC-AUC is about 0.045, so the differences between models are within noise. The test set has only 109 bad cases (bootstrap CI for LR ROC-AUC: 0.437 to 0.573).

- Best parameters: C = 0.3, L2 penalty, no class weight (the best CV PR-AUC is 0.0286, versus 0.028 for default LR).
- With the default 0.5 threshold the model never predicts "bad". A max-F1 threshold (0.0408, chosen on out-of-fold train data) flags 2.3% of test applicants and catches 6 of 109 bad cases.
- Decile lift on test is non-monotonic (top-risk decile lift 1.0x), and the Brier score is no better than predicting the base rate.
- Imbalance: SMOTE had the lowest ROC-AUC (0.545) and PR-AUC (0.022); class weights and undersampling were about equal to plain LR in ranking quality.
- Label sensitivity: with 30+ days overdue (11.8% bad), CV ROC-AUC is still only 0.527 (std 0.007), so the weak signal is not caused by the rarity of the 60+ label.

## Limitations
- `credit_record.csv` appears truncated, so many applicants have no label.
- Duplicate-profile grouping is an assumption; the dataset has no documentation on it.
- Odds ratios are measured against dropped reference categories, and some categories have very few bad cases. They should not be read as findings.
- Application data may simply not predict delinquency well; richer behavioral data would be needed.

## Predict function
`predict_approval(applicant_dict)` returns a risk probability and an APPROVE / REVIEW decision. It is a demonstration, not a production decision rule.

## Files
- `credit_card_approval.ipynb`: full notebook
- `figures/`: raw EDA, processed EDA, model evaluation
- `results/`: CV, grid-search, comparison, sensitivity tables, coefficients, test predictions, final summary
- `models/lr_best.joblib`: fitted pipeline
- `data_sample/`: first 200 rows of the cleaned data (full data is not included)
