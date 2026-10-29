import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

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

# 4. TRAIN RANDOM FOREST
rf_model = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestClassifier(n_estimators=100, random_state=42)
)

rf_model.fit(X_train, y_train)

# 5. FEATURE IMPORTANCE
rf = rf_model.named_steps["randomforestclassifier"]

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance", ascending=False
)

print("\n===== FEATURE IMPORTANCE =====")
print("\nTop 10 Important Features:")
print(importance.head(10).to_string(index=False))

# 6. FEATURE IMPORTANCE CHART
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