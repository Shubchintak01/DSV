import pandas as pd

# Creating a sample DataFrame
data = {'EmpID': [101, 102, 103, 104, 105],
    'Name': ['Aman', 'Priya', 'Rohit', 'Sneha', 'Vikram'],
    'Department': ['IT', 'HR', 'IT', 'Finance', 'HR'],
    'Salary': [65000, 48000, 72000, 58000, 52000],
    'Bonus': [5000, 3000, 6000, 4000, 3500]}

df = pd.DataFrame(data)

print(df)

print(df.head(3))

print(df.info())

# 1. Filtering rows
it_dept = df[df['Department'] == 'IT']

print(it_dept)

# 2. Creating a new column
df['Total_Comp'] = df['Salary'] + df['Bonus']

print(df)

# 3. Grouping and aggregation
dept_summary = df.groupby('Department')['Salary'].agg(['mean', 'max', 'count'])

print(dept_summary)

# 4. Sorting
sorted_df = df.sort_values(by='Total_Comp', ascending=False)

print("--- Department Summary ---")
print(dept_summary)
print("\n--- Sorted DataFrame ---")
print(sorted_df[['Name', 'Department', 'Total_Comp']])