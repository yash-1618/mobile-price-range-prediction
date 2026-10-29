# 📱 Mobile Price Range Prediction

<p align="center">
  <strong>Predictive Analytics Using Machine Learning</strong>
  <br>
  INT234 — Academic Project
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Machine%20Learning-Classification-orange" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Scikit--learn-Model%20Evaluation-F7931E?logo=scikitlearn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Project-Completed-success" alt="Project Status">
</p>

---

## 📌 Project Overview

Mobile phones have different technical specifications that influence their market prices. This project uses **predictive analytics and supervised machine learning** to classify mobile phones into one of four price ranges based on their hardware specifications.

The project involves data preparation, exploratory data analysis (EDA), feature engineering, classification model development, performance evaluation, and comparison of machine learning algorithms.

## 🎯 Project Objectives

- Understand the dataset and examine the distribution of mobile phone features.
- Analyze relationships between technical specifications and price ranges.
- Perform data cleaning and preprocessing.
- Apply feature engineering wherever necessary.
- Train and compare multiple supervised machine learning classification models.
- Evaluate model performance using appropriate classification metrics.
- Identify the best-performing model based on experimental results.
- Analyze prediction errors and interpret the results using visualizations.

## 📂 Dataset Description

**Dataset:** `Mobile Price Classification.csv`

The dataset contains technical specifications of mobile phones, including battery power, RAM, internal memory, camera specifications, screen dimensions, connectivity features, and other hardware characteristics.

**Target variable:** `price_range`

The target variable contains four classes:

| Class | Price Category |
|:---:|---|
| 0 | Low cost |
| 1 | Medium cost |
| 2 | High cost |
| 3 | Very high cost |

The objective is to predict the appropriate price category using the available mobile phone features.

## 🛠️ Technologies & Libraries

| Technology | Application |
|---|---|
| Python | Programming and analysis |
| Pandas | Data manipulation and preprocessing |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning and model evaluation |
| Visual Studio Code | Development environment |

## 🤖 Machine Learning Models

The project compares the following classification algorithms:

### 1. Logistic Regression
Used as a classification model to predict the price category from mobile phone specifications.

### 2. K-Nearest Neighbours (KNN)
Classifies a mobile phone based on the classes of its nearest observations in the feature space.

### 3. Gaussian Naive Bayes
Uses probability distributions and Bayes' theorem to predict the price category.

### 4. Decision Tree
Uses feature-based decision rules to classify mobile phones.

### 5. Support Vector Machine (SVM)
Identifies decision boundaries that separate the price categories.

### 6. Random Forest
Combines multiple decision trees to produce classification predictions.

## 📊 Evaluation Metrics

The models are evaluated using multiple performance metrics to provide a broader assessment of their predictive capabilities.

| Metric | Purpose |
|---|---|
| Accuracy | Measures the proportion of correctly classified observations |
| Precision | Measures the correctness of positive predictions |
| Recall | Measures how effectively the model identifies observations belonging to each class |
| F1-score | Combines precision and recall into a single measure |
| Log Loss | Evaluates the quality of predicted class probabilities |
| ROC-AUC | Measures the model's ability to distinguish between classes |

Additional evaluation includes a **confusion matrix** and a summary of misclassified observations to examine prediction errors.

## 📈 Exploratory Data Analysis & Visualizations

Exploratory data analysis helps identify patterns, distributions, and relationships within the dataset.

The project includes visualizations such as:

- Correlation heatmap of numerical features
- Battery power distribution by price range
- RAM distribution by price range
- Comparison of important features
- Model performance comparison
- Confusion matrix
- Feature importance analysis, where applicable

These visualizations help explain the dataset characteristics and support the interpretation of model performance.

## 📁 Repository Structure

```text
mobile-price-range-prediction/
│
├── README.md
├── Mobile Price Classification.csv
├── main.py
├── Data Overview and Cleaning.py
├── EDA_Correlation.py
├── Feature Engineering.py
├── Feature Importance.py
├── Model_Training_and_Comparison.py
├── Objectives and Conclusion.py
├── Visualization.py
│
└── images/
    ├── Battery Power by Price Range.png
    ├── Confusion Matrix.png
    ├── Correlation heatmap.png
    ├── Model Performance Comparison.png
    ├── RAM by Price Range.png
    └── Top 10.png
```

*Note: The structure above is illustrative. The actual repository should contain only the files you upload. Update the filenames and folder structure if your final files differ.*

## ⚙️ Installation & Execution

### Prerequisites

- Python 3
- Visual Studio Code or another Python development environment

### Step 1: Install the required libraries

Open a terminal and run:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

### Step 2: Obtain the project

Download the repository using GitHub's **Code → Download ZIP** option, or clone it using:

```bash
git clone https://github.com/yash-1618/mobile-price-range-prediction.git
```

### Step 3: Open the project

Extract the downloaded ZIP file if necessary, then open the project folder in Visual Studio Code.

Ensure that `Mobile Price Classification.csv` is in the location expected by the Python script.

### Step 4: Run the project

To execute the main project script, run:

```bash
python main.py
```

If you are using separate Python scripts for individual project stages, execute the appropriate script as required.

**Note:** The command assumes the main script is named `main.py` and that its dataset path and dependencies are configured correctly.

## 🏆 Results & Conclusion

This project demonstrates the application of supervised machine learning to mobile phone price range classification.

By comparing multiple algorithms and evaluating their performance using classification metrics, the project provides a structured approach to identifying an appropriate predictive model.

The use of exploratory visualizations, feature analysis, confusion matrices, and misclassification summaries also helps interpret the model's predictions and identify areas where classification errors occur.

The final model selection is based on the experimental results obtained during implementation.

## 🎓 Academic Information

| Detail | Information |
|---|---|
| Project Title | Mobile Price Range Prediction |
| Subject Code | INT234 |
| Subject | Predictive Analytics |
| Student Name | Saksham Rawat |
| Roll Number | 53 |
| Registration Number | 12408753 |
| Project Guide | Dr. Mrinalini Rana |

## 👨‍💻 Author

**Saksham Rawat**

INT234 — Predictive Analytics Project

---

<p align="center">
  <i>Exploring data-driven insights through machine learning and predictive analytics.</i>
</p>
