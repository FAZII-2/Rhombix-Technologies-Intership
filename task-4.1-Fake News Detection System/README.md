# Fake News Detection using Machine Learning

## Overview
This project builds a machine learning model that classifies news articles as **Real** or **Fake**. It covers the full pipeline: data collection, text cleaning, TF-IDF vectorization, model training, evaluation, and a simple interface where a user can paste a news headline and get an instant prediction.

## Dataset
- **Name:** Fake and Real News Dataset (Kaggle, by Clément Bisaillon)
- **Files:** `Fake.csv` and `True.csv`
- **Columns:** title, text, subject, date
- **After merging and removing duplicate articles:** 39,105 articles (21,197 Real, 17,908 Fake)

## Project Workflow
1. **Data loading and labeling:** Fake articles labeled `0`, Real articles labeled `1`. Title and text were merged into a single `content` column and duplicates were removed.
2. **Text preprocessing:**
   - Converted text to lowercase
   - Removed URLs, punctuation, numbers, and special characters
   - Removed the "(Reuters)" source tag to prevent the model from learning a shortcut instead of real patterns
   - Tokenization
   - Stop word removal
3. **Feature extraction:** TF-IDF vectorization (5,000 features, unigrams and bigrams), fitted on the training set only.
4. **Train/test split:** 80% training (31,284 samples) and 20% testing (7,821 samples), stratified by label.
5. **Model training:** Logistic Regression, Naive Bayes, and Random Forest were trained and compared.
6. **Evaluation:** Accuracy, precision, recall, F1-score, and a confusion matrix.
7. **Interface:** An interactive text box and a Predict button built with `ipywidgets`.

## Results

### Model Comparison
| Model | Accuracy |
|---|---|
| Logistic Regression | **98.31%** |
| Random Forest | 98.18% |
| Naive Bayes | 94.00% |

**Logistic Regression** gave the best accuracy and was selected as the final model.

### Classification Report (Logistic Regression)
| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Fake | 0.99 | 0.97 | 0.98 | 3,582 |
| Real | 0.98 | 0.99 | 0.98 | 4,239 |
| **Accuracy** | | | **0.98** | 7,821 |

### Confusion Matrix
|  | Predicted Fake | Predicted Real |
|---|---|---|
| **Actual Fake** | 3,489 | 93 |
| **Actual Real** | 39 | 4,200 |

- Correct predictions: **7,689**
- Wrong predictions: **132** (93 fake articles predicted as real, 39 real articles predicted as fake)

## Interface
The last cell of the notebook provides a text box where the user can paste a news headline or article. After clicking **Predict**, the model shows whether the news is **REAL** or **FAKE**, along with a confidence percentage.

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- ipywidgets
- Kaggle Notebook

## How to Run
1. Open the notebook in Kaggle.
2. Add the **Fake and Real News Dataset** as an input.
3. Run all cells in order.
4. Use the interface in the last cell to test your own headlines.

## Conclusion
The model reaches about 98% accuracy in separating real and fake news using TF-IDF features and Logistic Regression. The results are strong on this dataset, but this dataset has its own writing style, so performance on news from other sources may be lower. Using a more diverse dataset would be a good next step.

## Author
Faizan Ali
Internship Project: Rhombix Technologies, 
