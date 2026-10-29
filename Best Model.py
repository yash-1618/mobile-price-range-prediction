import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 1. LOAD DATA
df = pd.read_csv("Mobile Price Classification.csv")

df = df.drop_duplicates()
df = df.dropna(subset=["price_range"])

# 2. FEATURE ENGINEERING
df["px_area"] = df["px_height"] * df["px_width"]
df["screen_area"] = df["sc_h"] * df["sc_w"]

# 3. PREPARE DATA
X = df.drop("price_range", axis=1)
y = df["price_range"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. TRAIN BEST MODEL
best_model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)

best_model.fit(X_train, y_train)
best_predictions = best_model.predict(X_test)

# 5. CLASSIFICATION REPORT
print("\n===== BEST MODEL EVALUATION =====")
print("Best Model: Logistic Regression")

print("\nClassification Report:")
print(classification_report(
    y_test, best_predictions, zero_division=0
))

# 6. CONFUSION MATRIX
class_labels = [0, 1, 2, 3]

cm = confusion_matrix(
    y_test, best_predictions, labels=class_labels
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

plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Price Range")
plt.ylabel("Actual Price Range")
plt.tight_layout()
plt.show()

# 7. MISCLASSIFICATION ANALYSIS
print("\n===== MISCLASSIFICATION ANALYSIS =====")

actual = y_test.to_numpy()
wrong = best_predictions != actual

print("Correct Predictions:", int((~wrong).sum()))
print("Incorrect Predictions:", int(wrong.sum()))
print("Total Test Records:", len(y_test))
print("Misclassification Rate:", round(wrong.mean() * 100, 2), "%")

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
    print(wrong_predictions["Actual"].value_counts().sort_index())