import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Data setup ( we can use some data from API or Excel )
market_size_data = {
    "Year": [2024, 2025, 2026, 2027, 2028, 2029],
    "Market Size (USD Billion)": [20.02, 22.21, 24.64, 27.35, 30.39, 33.67]
}

units_data = {
    "Year": [2021, 2022, 2023, 2024],
    "Completed Units (Cairo)": [19000, 29000, 32000, 35000]  # Hypothetical values for 2023 and 2024
}

price_projection_data = {
    "Year": [2024, 2025, 2026, 2027, 2028, 2029],
    "Predicted Avg Price per m² (USD)": [800, 840, 880, 925, 970, 1020]
}

#  DataFrames convert
market_df = pd.DataFrame(market_size_data)
units_df = pd.DataFrame(units_data)
price_projection_df = pd.DataFrame(price_projection_data)

# Polynomial Regression for Price Projection
X_price = price_projection_df["Year"].values.reshape(-1, 1)
y_price = price_projection_df["Predicted Avg Price per m² (USD)"]
poly_features = np.polyfit(price_projection_df["Year"], y_price, 3)
poly_model = np.poly1d(poly_features)

# Create predictions
future_years = np.arange(2024, 2030)
price_predictions = poly_model(future_years)

# Set up subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Market Size Projection
axs[0, 0].plot(market_df["Year"], market_df["Market Size (USD Billion)"], marker='o', color='blue', label="Market Size")
axs[0, 0].set_title("Egypt Residential Real Estate Market Size (2024-2029)")
axs[0, 0].set_xlabel("Year")
axs[0, 0].set_ylabel("Market Size (USD Billion)")
axs[0, 0].grid()
axs[0, 0].legend()

# Units Completion till 2024
axs[0, 1].bar(units_df["Year"], units_df["Completed Units (Cairo)"], color='skyblue', label="Completed Units")
axs[0, 1].set_title("Units Completed in Cairo (2021-2024)")
axs[0, 1].set_xticks(units_df["Year"])
axs[0, 1].set_xlabel("Year")
axs[0, 1].set_ylabel("Units Completed")
axs[0, 1].grid()
axs[0, 1].legend()

# Projected Average Price per m² from 2024 onward
axs[1, 0].plot(future_years, price_predictions, linestyle='--', color='purple', label="Predicted Avg Price per m²")
axs[1, 0].set_title("Projected Avg Price per m² in Cairo (2024-2029)")
axs[1, 0].set_xlabel("Year")
axs[1, 0].set_ylabel("Avg Price per m² (USD)")
axs[1, 0].grid()
axs[1, 0].legend()

# Multi-Factor Combined Plot with Secondary Axis for Avg Price per m²
ax1 = axs[1, 1]
ax1.plot(market_df["Year"], market_df["Market Size (USD Billion)"], color='blue', marker='o', label="Market Size")
ax1.set_xlabel("Year")
ax1.set_ylabel("Market Size (USD Billion)", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

#  Units Completed on the same axis as Market Size
ax1.bar(units_df["Year"], units_df["Completed Units (Cairo)"], color='skyblue', alpha=0.5, label="Units Completed")

# Add a secondary y-axis for Avg Price per m²
ax2 = ax1.twinx()
ax2.plot(future_years, price_predictions, color='purple', linestyle='--', label="Predicted Avg Price per m²")
ax2.set_ylabel("Avg Price per m² (USD)", color='purple')
ax2.tick_params(axis='y', labelcolor='purple')

# Title for Combined Plot
ax1.set_title("Combined: Market Size, Units, and Projected Avg Price per m²")
fig.tight_layout()
plt.show()
