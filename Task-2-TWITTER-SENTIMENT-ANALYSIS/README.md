# Twitter Sentiment Analysis

A complete, end-to-end sentiment analysis system built on the Sentiment140 dataset (1.6M tweets), covering classical NLP feature engineering, multi-model comparison (Naive Bayes, Logistic Regression, SVM, LSTM), model interpretability, a confidence-based Neutral-sentiment enhancement, and an interactive topic-level sentiment demo.

Built as part of the CodeAlpha Machine Learning Internship.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Text Preprocessing](#text-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Models & Results](#models--results)
- [Model Interpretability](#model-interpretability)
- [Enhancement: Confidence-Based Neutral Zone](#enhancement-confidence-based-neutral-zone)
- [Topic-Level Sentiment Demo](#topic-level-sentiment-demo)
- [Limitations & Honest Tradeoffs](#limitations--honest-tradeoffs)
- [How to Run](#how-to-run)
- [Repository Structure](#repository-structure)
- [Possible Future Improvements](#possible-future-improvements)
- [Author](#author)

---

## Project Overview

**Goal:** Build a sentiment analysis model for Twitter data that can classify tweets as Positive or Negative, and analyze public sentiment on specific topics or keywords.

**What this project demonstrates:**
- A full, production-style NLP preprocessing pipeline
- Comparison of four different modeling approaches — from simple probabilistic models to deep learning
- Model interpretability (understanding *why* the model predicts what it predicts)
- An original enhancement (confidence-based Neutral zone) that improves on the raw binary labels available in the source data
- A working, interactive topic-sentiment tool — the actual deliverable requested by the project brief

---

## Dataset

**Source:** [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) (Kaggle)

- **Size:** 1.6 million tweets
- **Labels:** Binary — Negative (0) and Positive (4, remapped to 1)
- **Label origin:** Weakly supervised — labels were auto-assigned based on emoticons present in the original tweet (e.g. `:)` → Positive, `:(` → Negative), then emoticons were stripped from the text
- **Class balance:** Perfectly balanced — 800,000 Negative / 800,000 Positive

**Why this dataset:** Its scale and topic diversity make it suitable for training a genuinely robust, general-purpose sentiment classifier that generalizes across topics — rather than a narrow, domain-specific model.

**Known limitations:**
- No Neutral class exists in the original labels (addressed later via a confidence-based enhancement)
- Labels are derived from emoticons, not human annotation — introducing some label noise (e.g., sarcasm is not accounted for)

---

## Project Workflow

1. Data acquisition (Kaggle via `kagglehub`)
2. Data loading & label cleanup
3. Exploratory Data Analysis — before preprocessing
4. Text preprocessing / cleaning
5. Exploratory Data Analysis — after preprocessing
6. Feature engineering (TF-IDF, unigrams + bigrams)
7. Model training — Logistic Regression, Naive Bayes, Linear SVM, LSTM
8. Model evaluation & comparison
9. Model interpretability analysis
10. Confidence-based Neutral zone enhancement
11. Interactive topic-level sentiment demo

---

## Exploratory Data Analysis

EDA was performed both **before and after** text cleaning, to visualize the impact of preprocessing:

**Before preprocessing:**
- Class balance
- Tweet character/word length distribution by sentiment
- Most common raw words and hashtags
- Punctuation intensity (`!` count) by sentiment
- Word cloud of raw tweet text

**After preprocessing:**
- Cleaned word count distribution
- Before vs. after average word count comparison
- Most common cleaned words, per sentiment class
- Sentiment-specific word clouds (Positive vs. Negative)
- Top bigrams — validating the choice to use bigram features downstream

---

## Text Preprocessing

Each tweet passes through the following cleaning pipeline:

1. Lowercasing
2. Contraction expansion (e.g. `"can't"` → `"cannot"`)
3. URL removal
4. `@mention` removal
5. Hashtag normalization (`#great` → `great` — keeps the word, drops the symbol)
6. Repeated character reduction (`"sooooo"` → `"soo"`)
7. Punctuation & digit removal
8. Stopword removal
9. Lemmatization

Rows that became empty after cleaning (e.g. tweets that were only a URL or mention) were dropped from the dataset.

---

## Feature Engineering

- **TF-IDF vectorization** with:
  - Unigrams + bigrams (`ngram_range=(1,2)`) — captures short phrases like *"not good"* or *"cannot wait"* that unigrams alone would miss
  - Vocabulary capped at 50,000 terms
  - Minimum document frequency of 5, to filter out rare/noisy terms
- **Train/test split performed before vectorization**, to avoid data leakage — the vectorizer is fit only on training data

For the LSTM model, a separate pipeline was used: tweets were tokenized into integer sequences (vocabulary capped at 30,000 words) and padded/truncated to a fixed length of 40 tokens, since LSTMs require ordered sequences rather than bag-of-words vectors.

---

## Models & Results

Four models were trained and evaluated on an identical held-out test set (20% of the data, stratified by class):

| Model | Test Accuracy |
|---|---|
| **Logistic Regression** | **0.7902** |
| Linear SVM | 0.7866 |
| LSTM | 0.7847 |
| Multinomial Naive Bayes | 0.7731 |

**Key finding:** The classical TF-IDF + Logistic Regression baseline outperformed the LSTM in this run. This is a legitimate, explainable result rather than a modeling error — the LSTM was trained for only 5 epochs (a deliberate compute-time tradeoff on Colab's free GPU tier), and its validation loss was still decreasing when training was stopped. With more training time, the LSTM would likely close or exceed this gap. This result is reported transparently rather than hidden, since compute-constrained model comparison is itself a realistic engineering scenario.

All four models were evaluated using accuracy, per-class precision/recall/F1 (via `classification_report`), and confusion matrices — not accuracy alone — since class-level performance matters for a fair, complete assessment.

---

## Model Interpretability

For the Logistic Regression model, feature coefficients were inspected to identify the words/phrases most strongly associated with each sentiment class.

**Top Positive drivers included:** `thank`, `welcome`, `congrats`, `smile`, `cannot wait`
**Top Negative drivers included:** `sad`, `bummed`, `gutted`, `disappointing`, `cannot`

Notably, `cannot wait` ranked as strongly *Positive* despite containing the word `cannot` (which alone is strongly *Negative*) — direct evidence that the bigram feature engineering choice successfully captured context that unigram-only features would have missed.

---

## Enhancement: Confidence-Based Neutral Zone

Sentiment140 provides only binary labels, forcing every tweet into Positive or Negative — even genuinely ambiguous or neutral ones (e.g., *"The train arrives at 5pm"*).

**Approach:** Instead of using the model's hard 0/1 prediction, its underlying probability score (`predict_proba`) is used to define a confidence band:

- `P(Positive) < 0.40` → Negative
- `0.40 ≤ P(Positive) ≤ 0.60` → **Neutral**
- `P(Positive) > 0.60` → Positive

**Result:** 15.7% of test tweets (49,838 of 318,208) fell into the low-confidence Neutral zone — tweets that were previously force-labeled Positive or Negative despite the model having near-coin-flip confidence.

**Key property:** This required **no retraining**. It reuses the already-trained model's existing probability outputs — a fast, practical technique for adding a Neutral category to a binary-only labeled dataset.

*Note: the 0.40/0.60 threshold is a reasonable, documented judgment call rather than a value derived analytically from the data.*

---

## Topic-Level Sentiment Demo

An interactive function allows querying sentiment for any topic or keyword:

```python
user_topic = input("Enter a topic/keyword to analyze sentiment for: ")
result = analyze_topic_sentiment_v2(user_topic)
```

Given a topic (e.g. a brand, hashtag, or event name), it:
1. Filters all tweets mentioning that keyword
2. Classifies each into Positive / Neutral / Negative using the confidence-based logic above
3. Displays a percentage breakdown and a bar chart
4. Surfaces the most confidently Positive and most confidently Negative example tweets

This directly fulfills the original project brief: *"Analyze tweets to understand public sentiment on specific topics."*

---

## Limitations & Honest Tradeoffs

This project makes a few deliberate, documented tradeoffs rather than hiding them:

- **Lemmatization without POS tagging:** `WordNetLemmatizer` defaults to noun-based lemmatization when no part-of-speech is specified. This means some verb forms (e.g. *"texting"*) are not reduced to their root (*"text"*). Adding POS tagging would fix this at the cost of significantly more preprocessing time across 1.6M rows.
- **LSTM trained for only 5 epochs:** Due to Colab compute-time constraints, LSTM training was capped before full convergence. Validation loss was still improving at the stopping point.
- **No true Neutral labels:** The Neutral zone is a post-hoc confidence-based heuristic, not a class the model was ever trained to recognize directly.
- **Emoticon-derived labels:** Sentiment140's labels were assigned automatically based on emoticons, not human review, so some label noise (e.g. sarcasm) is inherently present in training data.

---

## How to Run

1. Open the notebook in **Google Colab**
2. (Recommended) Set runtime to **T4 GPU**: `Runtime → Change runtime type → T4 GPU`
3. Run all cells top to bottom — the notebook is fully self-contained and documented with markdown explanations above every code cell
4. A Kaggle API token may be required on first run to download the dataset via `kagglehub` (create one at Kaggle → Account Settings → Create New API Token)
5. For the interactive demo cell, simply type any keyword when prompted

---

## Repository Structure

```
twitter-sentiment-analysis/
│
├── README.md                     <- You are here
├── Twitter_Sentiment_Analysis.ipynb   <- Full Colab notebook (all phases)
└── requirements.txt               <- (optional) Python dependencies for local runs
```

---

## Possible Future Improvements

- Train the LSTM to full convergence (more epochs) or fine-tune a transformer model (e.g. DistilBERT) for a stronger deep learning comparison point
- Add POS-aware lemmatization for cleaner text normalization
- Deploy the topic-sentiment tool as a small Streamlit/Gradio web app
- Incorporate true human-labeled Neutral examples (e.g. from a 3-class dataset) to train a native 3-class classifier instead of using a post-hoc confidence heuristic

---

## Author

**Faizan**
CodeAlpha Machine Learning Internship
