import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Generate sample data
np.random.seed(42)
x = np.random.normal(loc=50, scale=10, size=500)
y = 2.5 * x + np.random.normal(loc=0, scale=15, size=500)

df = pd.DataFrame({'Feature_X': x, 'Feature_Y': y})

print(df.head())

print(df.tail())

print(df.info())

print(df.describe())

# Statistics
mean_x = np.mean(x)
median_x = np.median(x)
variance_x = np.var(x, ddof=1)
correlation = df.corr()

print(f"Mean: {mean_x:.2f} | Median: {median_x:.2f} | Variance: {variance_x:.2f}")
print("\nCorrelation Matrix:\n", correlation)

# Visualizations
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 1. Initialize the grid
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 2. First plot: Heatmap
sns.heatmap(correlation, annot=True, cmap="coolwarm", ax=axes[0])
axes[0].set_title("Correlation Heatmap")

# 3. Second plot: Distribution
sns.histplot(x, kde=False, stat="density", color="skyblue", ax=axes[1])
xmin, xmax = axes[1].get_xlim()
x_axis = np.linspace(xmin, xmax, 100)
axes[1].plot(
    x_axis,
    stats.norm.pdf(x_axis, mean_x, np.sqrt(variance_x)),
    "r-",
    lw=2,
    label="Normal Fit",
)
axes[1].set_title("Normal Distribution of Feature X")
axes[1].legend()

# 4. Render
plt.tight_layout()
plt.show()