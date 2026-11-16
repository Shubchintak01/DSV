import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Export dummy data to CSV for demonstration
sample_data = {
    'ID': [1, 2, 3, 4, 5],
    'Age': [23, 45, np.nan, 35, 29],
    'Score': [88, 92, 79, np.nan, 85],
    'Experience': [1, 15, 6, 8, 3]
}

pd.DataFrame(sample_data).to_csv('students.csv', index=False)

# Read CSV
df = pd.read_csv('students.csv')

# Exploration
print("1. Shape:", df.shape)

print("\n2. Head:\n", df.head())

print("\n3. Missing values:\n", df.isnull().sum())

print("\n4. Summary stats:\n", df.describe())

print("\n5. Correlation matrix:\n", df.corr(numeric_only=True))

# Correct
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

plt.show()

# Fill missing values and export back
df_cleaned = df.fillna(df.mean(numeric_only=True))

print(df_cleaned)

df_cleaned.to_csv('students_cleaned.csv', index=False)