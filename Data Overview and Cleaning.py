
import pandas as pd


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