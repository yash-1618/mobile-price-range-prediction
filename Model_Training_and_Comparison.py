
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
    roc_auc_score
)


# ==============================
# LOAD DATA AND REQUIRED FEATURES
# ==============================

df = pd.read_csv("Mobile Price Classification.csv")

df = df.drop_duplicates()
df = df.dropna(subset=["price_range"])

# Approximate total pixel count
df["px_area"] = df["px_height"] * df["px_width"]

# Approximate screen area
df["screen_area"] = df["sc_h"] * df["sc_w"]


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
