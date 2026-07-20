"""
DAY 2: Pandas Fundamentals
Pandas = Panel Data - Excel on steroids!
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("🐼 PANDAS - DATA MANIPULATION MADE EASY")
print("=" * 60)

# ========================================
# 1. CREATING DATAFRAMES (Excel Sheets)
# ========================================

print("\n📊 1. CREATING DATAFRAMES")

# Create a DataFrame from a dictionary
data = {
    'Name': ['Arthur', 'Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 40, 28],
    'City': ['NYC', 'LA', 'Chicago', 'NYC', 'LA'],
    'Salary': [75000, 85000, 95000, 110000, 68000],
    'Skill_Level': [3, 4, 5, 5, 2]  # 1-5 scale
}

df = pd.DataFrame(data)
print("Our DataFrame:")
print(df)

# ========================================
# 2. VIEWING DATA
# ========================================

print("\n👀 2. VIEWING DATA")

print(f"Shape: {df.shape}")  # (rows, columns)
print(f"Columns: {df.columns.tolist()}")
print(f"Data types:\n{df.dtypes}")

print("\nFirst 3 rows:")
print(df.head(3))

print("\nLast 2 rows:")
print(df.tail(2))

print("\nBasic statistics:")
print(df.describe())

# ========================================
# 3. SELECTING DATA
# ========================================

print("\n🎯 3. SELECTING DATA")

# Select a column
print("Selecting a column:")
print(df['Name'])
print(df.Name)  # Alternative

# Select multiple columns
print("\nSelecting multiple columns:")
print(df[['Name', 'City']])

# Select rows by index
print("\nRow at index 2:")
print(df.iloc[2])  # By position

print("\nRows 1-3:")
print(df.iloc[1:4])

# Select by condition (filtering)
print("\nPeople with Salary > 80000:")
print(df[df['Salary'] > 80000])

print("\nPeople from NYC:")
print(df[df['City'] == 'NYC'])

print("\nPeople with Skill_Level >= 4:")
print(df[df['Skill_Level'] >= 4])

# ========================================
# 4. MODIFYING DATA
# ========================================

print("\n✏️ 4. MODIFYING DATA")

# Add a new column
df['Bonus'] = df['Salary'] * 0.1  # 10% bonus
print("After adding Bonus column:")
print(df)

# Update values
df.loc[df['City'] == 'NYC', 'City'] = 'New York'
print("\nAfter updating NYC to New York:")
print(df)

# Add a calculated column
df['Salary_Per_Year'] = df['Salary'] * 12
print("\nAfter adding Salary_Per_Year:")
print(df)

# ========================================
# 5. GROUPING & AGGREGATION
# ========================================

print("\n📈 5. GROUPING & AGGREGATION")

# Group by City
print("Average salary by city:")
city_stats = df.groupby('City')['Salary'].mean()
print(city_stats)

print("\nMultiple statistics by city:")
city_summary = df.groupby('City').agg({
    'Salary': ['mean', 'min', 'max', 'count'],
    'Skill_Level': ['mean', 'max']
})
print(city_summary)

# ========================================
# 6. SORTING & RANKING
# ========================================

print("\n🔢 6. SORTING & RANKING")

print("Sorted by Salary (descending):")
print(df.sort_values('Salary', ascending=False))

print("\nSorted by Age (ascending):")
print(df.sort_values('Age'))

# ========================================
# 7. HANDLING MISSING DATA
# ========================================

print("\n🔧 7. HANDLING MISSING DATA")

# Create data with missing values
df_with_na = df.copy()
df_with_na.loc[1, 'Salary'] = np.nan  # Add NaN
df_with_na.loc[3, 'Skill_Level'] = np.nan

print("Data with missing values:")
print(df_with_na)

print(f"\nMissing values count:")
print(df_with_na.isnull().sum())

# Fill missing values
df_filled = df_with_na.fillna({
    'Salary': df_with_na['Salary'].mean(),
    'Skill_Level': df_with_na['Skill_Level'].median()
})
print("\nFilled missing values:")
print(df_filled)

# ========================================
# 8. YOUR FIRST DATA SCIENCE OPERATION
# ========================================

print("\n🤖 8. YOUR FIRST DATA SCIENCE OPERATION")

# Calculate correlation between numeric columns
numeric_cols = df.select_dtypes(include=[np.number])
correlation = numeric_cols.corr()
print("Correlation matrix:")
print(correlation)

print("\nInsights:")
print(f"Age vs Salary correlation: {correlation.loc['Age', 'Salary']:.3f}")
print(f"Skill_Level vs Salary correlation: {correlation.loc['Skill_Level', 'Salary']:.3f}")

# Create a summary
print("\n📊 Data Summary:")
print(f"Total people: {len(df)}")
print(f"Average age: {df['Age'].mean():.1f}")
print(f"Average salary: ${df['Salary'].mean():,.0f}")
print(f"Average skill level: {df['Skill_Level'].mean():.1f}")