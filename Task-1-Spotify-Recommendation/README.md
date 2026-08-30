# Task 1 — Spotify Recommendation System

## 📌 Overview

This project was completed as part of my Machine Learning internship at Rhombix Technologies.
The project uses Spotify audio features and machine learning classification algorithms to predict whether a song is likely to be liked by a user.

---

## 🎯 Objective

The objective is to predict the `liked` value of a song:
`0` → Not Liked
`1` → Liked
The prediction is based on Spotify audio features such as danceability, energy, loudness, acousticness, instrumentalness, tempo, and others.

---

## 📊 Dataset

The dataset contains:
- **195 songs**
- **14 columns**
- **13 input features**
- **1 target variable**

### Features

- Danceability
- Energy
- Key
- Loudness
- Mode
- Speechiness
- Acousticness
- Instrumentalness
- Liveness
- Valence
- Tempo
- Duration
- Time Signature

### Target

`liked`

---

## 🔍 Data Understanding & EDA

The following analysis was performed:

- Dataset structure and data types
- Missing-value analysis
- Duplicate-value analysis
- Target distribution
- Feature distributions
- Correlation analysis
- Feature-to-target analysis
- Data visualizations

The target classes were found to be reasonably balanced.

---

## ⚙️ Data Preprocessing

The following steps were performed:

1. Feature and target separation
2. Train-test split
3. Feature scaling where required
4. Machine learning preprocessing pipeline

---

## 🤖 Machine Learning Models

Multiple classification algorithms were implemented and evaluated, including:

- K-Nearest Neighbors (KNN)
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boost

---

## 🔄 Machine Learning Pipeline

The project uses a Scikit-learn pipeline to organize preprocessing and model training.

General workflow:

```text
Raw Data
   ↓
Feature / Target Separation
   ↓
Train-Test Split
   ↓
Preprocessing
   ↓
Feature Scaling
   ↓
Machine Learning Model
   ↓
Prediction
   ↓
Evaluation
