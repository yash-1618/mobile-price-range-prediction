
import pandas as pd

df = pd.read_csv("Mobile Price Classification.csv")

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
