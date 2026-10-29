import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ==============================
# 1. LOAD DATA
# ==============================

df = pd.read_csv("Mobile Price Classification.csv")

print("\n===== DATA OVERVIEW =====")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nData Types:")
print(df.dtypes)

print("\nSummary Statistics:")
print(df.describe())


# ==============================
# 2. DATA CLEANING
# ==============================

print("\n===== DATA CLEANING =====")

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

print("\nDuplicate Records:", df.duplicated().sum())

# Remove duplicate records and rows without a target value
df = df.drop_duplicates()
df = df.dropna(subset=["price_range"])

print("\nDataset Shape After Cleaning:", df.shape)

print("Missing Target Values:", int(df["price_range"].isnull().sum()))

print(
    "Missing Predictor Values:",
    int(df.drop(columns=["price_range"]).isnull().sum().sum())
)

print(
    "Any remaining missing predictor values will be handled "
    "inside the model pipelines using training data only."
)


# ==============================
# 3. SUSPICIOUS ZERO VALUES
# ==============================

print("\n===== ZERO VALUE ANALYSIS =====")

zero_columns = ["fc", "pc", "sc_w", "px_height"]

zero_counts = (df[zero_columns] == 0).sum()

zero_percentages = (
    (df[zero_columns] == 0).mean() * 100
).round(2)

zero_summary = pd.DataFrame({
    "Zero Count": zero_counts,
    "Percentage (%)": zero_percentages
})

print("\nZero Values and Percentages:")
print(zero_summary)

print("\nZero values have been investigated and retained.")


# ==============================
# 4. FEATURE ENGINEERING
# ==============================

# Approximate total pixel count
df["px_area"] = df["px_height"] * df["px_width"]

# Approximate screen area
df["screen_area"] = df["sc_h"] * df["sc_w"]

print("\n===== FEATURE ENGINEERING =====")
print("Created px_area and screen_area.")


# ==============================
# 5. EXPLORATORY DATA ANALYSIS
# ==============================

print("\n===== EXPLORATORY DATA ANALYSIS =====")


# Objective 1: Price range distribution

print("\nObjective 1: Price Range Distribution")

class_counts = df["price_range"].value_counts().sort_index()

class_percentages = (
    df["price_range"].value_counts(normalize=True).sort_index() * 100
).round(2)

class_summary = pd.DataFrame({
    "Count": class_counts,
    "Percentage (%)": class_percentages
})

print(class_summary)


# Objective 2: Average specifications by price range

print("\nObjective 2: Average Specifications by Price Range")

print(
    df.groupby("price_range")
    .mean(numeric_only=True)
    .round(2)
)


# Objective 3: Correlation with price range

print("\nObjective 3: Correlation with Price Range")

correlations = (
    df.corr(numeric_only=True)["price_range"]
    .drop("price_range")
    .sort_values(ascending=False)
)

print(correlations.round(3))


# ==============================
# 6. VISUALIZATION
# ==============================

sns.set_theme(style="whitegrid")


# Chart 1: RAM by price range

plt.figure()
sns.boxplot(
    data=df,
    x="price_range",
    y="ram",
    color="steelblue"
)
plt.title("RAM by Price Range")
plt.xlabel("Price Range")
plt.ylabel("RAM")
plt.tight_layout()
plt.show()


# Chart 2: Battery power by price range

plt.figure()
sns.boxplot(
    data=df,
    x="price_range",
    y="battery_power",
    color="steelblue"
)
plt.title("Battery Power by Price Range")
plt.xlabel("Price Range")
plt.ylabel("Battery Power")
plt.tight_layout()
plt.show()


# Chart 3: Correlation heatmap

plt.figure(figsize=(12, 8))

sns.heatmap(
    df.corr(numeric_only=True),
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# ==============================
# 7. PREPARE DATA
# ==============================

print("\n===== DATA PREPARATION =====")

X = df.drop("price_range", axis=1)
y = df["price_range"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))


# ==============================
# 8. MACHINE LEARNING MODELS
# ==============================

print("\n===== MODEL TRAINING =====")

models = {

    "Logistic Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    ),

    "K-Nearest Neighbors": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    ),

    "Naive Bayes": make_pipeline(
        SimpleImputer(strategy="median"),
        GaussianNB()
    ),

    "Decision Tree": make_pipeline(
        SimpleImputer(strategy="median"),
        DecisionTreeClassifier(random_state=42)
    ),

    "Support Vector Machine": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVC(probability=True, random_state=42)
    ),

    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    )
}


results = []

class_labels = sorted(y.unique())


for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    loss = log_loss(
        y_test,
        probabilities,
        labels=class_labels
    )

    auc = roc_auc_score(
        y_test,
        probabilities,
        labels=class_labels,
        multi_class="ovr",
        average="macro"
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Log Loss": loss,
        "ROC-AUC": auc
    })

    print("\nModel:", name)
    print("Accuracy:", round(accuracy, 4))
    print("Macro Precision:", round(precision, 4))
    print("Macro Recall:", round(recall, 4))
    print("Macro F1:", round(f1, 4))
    print("Log Loss:", round(loss, 4))
    print("ROC-AUC:", round(auc, 4))


# ==============================
# 9. MODEL COMPARISON
# ==============================

print("\n===== MODEL COMPARISON =====")

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
).reset_index(drop=True)

print(results_df.round(4).to_string(index=False))


# Model comparison chart

plt.figure(figsize=(9, 5))

sns.barplot(
    data=results_df,
    x="F1 Score",
    y="Model",
    color="steelblue"
)

plt.title("Model Performance Comparison")
plt.xlabel("Macro F1 Score")
plt.ylabel("Model")
plt.tight_layout()
plt.show()


# ==============================
# 10. BEST MODEL EVALUATION
# ==============================

print("\n===== BEST MODEL EVALUATION =====")

best_name = results_df.iloc[0]["Model"]

best_model = models[best_name]

best_predictions = best_model.predict(X_test)

print("Best Model:", best_name)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        best_predictions,
        labels=class_labels,
        zero_division=0
    )
)


# Confusion matrix

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=class_labels
)

plt.figure()

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels
)

plt.title("Confusion Matrix - " + best_name)
plt.xlabel("Predicted Price Range")
plt.ylabel("Actual Price Range")
plt.tight_layout()
plt.show()


# ==============================
# 11. MISCLASSIFICATION ANALYSIS
# ==============================

print("\n===== MISCLASSIFICATION ANALYSIS =====")

actual = y_test.to_numpy()

wrong = best_predictions != actual

print("Correct Predictions:", int((~wrong).sum()))
print("Incorrect Predictions:", int(wrong.sum()))
print("Total Test Records:", len(y_test))

print(
    "Misclassification Rate:",
    round(wrong.mean() * 100, 2),
    "%"
)

wrong_predictions = pd.DataFrame({
    "Actual": actual[wrong],
    "Predicted": best_predictions[wrong]
})

print("\nIncorrect Predictions (Actual, Predicted):")

if wrong_predictions.empty:

    print("No incorrect predictions.")

else:

    print(wrong_predictions.value_counts().to_string())

    print("\nIncorrect Predictions by Actual Price Range:")

    print(
        wrong_predictions["Actual"]
        .value_counts()
        .sort_index()
    )


# ==============================
# 12. FEATURE IMPORTANCE
# ==============================

print("\n===== FEATURE IMPORTANCE =====")

rf = models["Random Forest"].named_steps[
    "randomforestclassifier"
]

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")

print(importance.head(10).to_string(index=False))


# Feature importance chart

plt.figure(figsize=(9, 5))

sns.barplot(
    data=importance.head(10),
    x="Importance",
    y="Feature",
    color="steelblue"
)

plt.title("Top 10 Important Features (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# ==============================
# 13. PROJECT OBJECTIVES
# ==============================

print("\n===== PROJECT OBJECTIVES =====")

print("1. Analyse the distribution of the four price classes.")

print("2. Compare mobile specifications across price classes.")

print(
    "3. Investigate relationships between specifications "
    "and price range."
)

print(
    "4. Train and compare six classification models "
    "using evaluation metrics."
)

print(
    "5. Analyse model errors and identify important "
    "features and findings."
)


# ==============================
# 14. CONCLUSION
# ==============================

print("\n===== PROJECT CONCLUSION =====")

print("Mobile price ranges were predicted using machine learning.")

print("Six classification models were compared on the same test set.")

print("The best model was selected using Macro F1 Score.")

print("Best Model:", best_name)

print(
    "Accuracy:",
    round(results_df.iloc[0]["Accuracy"], 4)
)

print(
    "Macro F1 Score:",
    round(results_df.iloc[0]["F1 Score"], 4)
)

print(
    "Log Loss:",
    round(results_df.iloc[0]["Log Loss"], 4)
)

print(
    "ROC-AUC:",
    round(results_df.iloc[0]["ROC-AUC"], 4)
)

print(
    "Most Important Feature:",
    importance.iloc[0]["Feature"]
)