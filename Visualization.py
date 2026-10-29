
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Mobile Price Classification.csv")


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
