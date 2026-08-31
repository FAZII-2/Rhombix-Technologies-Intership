# Facial Image Recognition & XAI Pipeline
**Rhombix Technologies Internship - Task 1.1**  
**Author:** Faizan Ali

## Project Overview
This repository contains a production-ready, end-to-end Machine Learning pipeline designed to classify human faces using the **Labeled Faces in the Wild (LFW)** dataset. 

The project demonstrates a complete Deep Learning lifecycle: streaming cloud data, curating datasets, building custom Convolutional Neural Networks (CNNs), implementing Transfer Learning (MobileNetV2), and finally deploying Explainable AI (XAI) via Grad-CAM to visually interpret the model's mathematical decision-making process.

---

## Tech Stack & Libraries
*   **Deep Learning Framework:** TensorFlow 2.x, Keras
*   **Pre-trained Architecture:** MobileNetV2 (ImageNet weights)
*   **Data Processing:** `tf.data` API, NumPy, Kaggle API
*   **Computer Vision:** Keras Preprocessing, OpenCV Implicit via Keras
*   **Statistical Evaluation:** Scikit-Learn (Classification Report, Confusion Matrix)
*   **Data Visualization:** Matplotlib, Seaborn

---

## Dataset & Curation
*   **Source:** Labeled Faces in the Wild (LFW) via Kaggle.
*   **Cloud Streaming:** Bypassed local storage constraints by downloading and extracting the dataset directly into the ephemeral Colab `/content/` directory.
*   **Threshold Filtering:** Raw data was heavily imbalanced. An automated Python script filtered the dataset to only include individuals with **>= 40 images**. 
*   **Final Classes:** 19 distinct individuals (classes).
*   **Validation Split:** Applied a strict **80% Training / 20% Validation split** via `image_dataset_from_directory` to ensure the model was evaluated purely on unseen data (optimizing GPU memory over standard K-Fold Cross-Validation).

---

## Project Workflow (Phases 1 to 5)

### Phase 1: Environment & Data Pipeline
*   **Determinism:** Set global `tf.random.set_seed` to ensure reproducible results.
*   **Data Augmentation:** Injected a `Sequential` augmentation layer inside the pipeline applying `RandomFlip("horizontal")`, `RandomRotation(0.1)`, and `RandomZoom(0.1)`. This artificially expanded the dataset to combat overfitting.
*   **Optimization:** Utilized `.cache()` and `.prefetch(tf.data.AUTOTUNE)` to dynamically load images into GPU memory without CPU bottlenecking.

### Phase 2: Custom Baseline CNN
*   Constructed a lightweight Convolutional Neural Network from scratch to establish a baseline metric.
*   **Architecture:** 3x Conv2D/MaxPooling blocks $\rightarrow$ Flatten $\rightarrow$ Dense(128) $\rightarrow$ Dropout(0.5) $\rightarrow$ Dense(19, Softmax).
*   **Performance:** Achieved **74.26% Validation Accuracy** and **0.93 Validation Loss**.

### Phase 3: Transfer Learning (MobileNetV2)
Transitioned to Google's MobileNetV2 to leverage advanced feature extraction trained on 1.4 million ImageNet images.
*   **Preprocessing:** Applied `tf.keras.applications.mobilenet_v2.preprocess_input` to zero-center pixel arrays to `[-1.0, 1.0]`.
*   **Compression:** Replaced traditional `Flatten()` with `GlobalAveragePooling2D` to prevent parameter explosion.
*   **Stage 1 (Feature Extraction):** Froze the 2.2M base parameters. Trained only the custom dense head for 15 epochs. Reached **72.12% Validation Accuracy**.
*   **Stage 2 (Fine-Tuning):** Unfroze the top 30 convolutional layers. Re-compiled with a microscopic learning rate (`1e-5`) to gently adapt the pre-trained weights to facial topography. 
*   **Final Performance:** Jumped to **79.62% Validation Accuracy** and **0.58 Validation Loss**.

### Phase 4: Statistical Evaluation
*   **Callbacks Used:** `EarlyStopping` (patience: 3[CNN] and 5[MobileNetV2], restore_best_weights: True) and `ModelCheckpoint`.
*   **Metrics:** Generated loss and accuracy curve visualizations across both training stages.
*   **Forensics:** Evaluated the unseen testing dataset using Scikit-Learn to output a detailed Precision/Recall/F1-Score report and a Seaborn heatmap Confusion Matrix to identify specific class collisions.

### Phase 5: Inference & Explainable AI (Grad-CAM)
*   **Real-Time Inference:** Built a Python function to ingest a single raw JPEG, dynamically reshape it to `(1, 224, 224, 3)`, apply MobileNet preprocessing, and output a confidence prediction.
*   **Grad-CAM (Gradient-weighted Class Activation Mapping):** 
    *   Digitally decoupled the MobileNet base from the custom classification head.
    *   Utilized `tf.GradientTape()` to calculate the partial derivatives (gradients) of the winning class score with respect to the 7x7 spatial feature map.
    *   Generated an RGB thermal heatmap and superimposed it over the original image at 40% opacity, visually proving exactly which facial landmarks triggered the neural network's mathematical decision.

---

## Final Model Metrics
| Metric | Baseline CNN | MobileNetV2 (Fine-Tuned) |
| :--- | :--- | :--- |
| **Training Accuracy** | 68.27% | 85.41% |
| **Validation Accuracy** | 74.26% | 79.62% |
| **Validation Loss** | 0.9365 | 0.5852 |

## How to Run
1. Open the `.ipynb` file in Google Colab.
2. Ensure the hardware accelerator is set to **T4 GPU**.
3. Provide your Kaggle `kaggle.json` credentials in the first cell to authenticate the dataset download.
4. Click **Run All**.
