
import pandas as pd


df = pd.read_csv("Mobile Price Classification.csv")


# ==============================
# 4. FEATURE ENGINEERING
# ==============================

# Approximate total pixel count
df["px_area"] = df["px_height"] * df["px_width"]

# Approximate screen area
df["screen_area"] = df["sc_h"] * df["sc_w"]

print("\n===== FEATURE ENGINEERING =====")
print("Created px_area and screen_area.")
