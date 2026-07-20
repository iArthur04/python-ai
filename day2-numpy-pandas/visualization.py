"""
DAY 2: Extended Data Visualization
Creates several different chart types from the 500-student dataset.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent
DATA_PATH = OUTPUT_DIR / "student_data_500.csv"
PASS_MARK = 60

print("=" * 70)
print("📊 EXTENDED STUDENT DATA VISUALIZATION")
print("=" * 70)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"{DATA_PATH.name} was not found. Run data_analysis.py before visualization.py."
    )

df = pd.read_csv(DATA_PATH)
df = pd.read_csv('student_data_500.csv')
chart_paths = []


def save_chart(filename: str) -> None:
    """Save the current figure and close it to avoid overlapping charts."""
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    chart_paths.append(path)
    print(f"✅ Saved {filename}")


# ========================================
# 1. HISTOGRAM — SCORE DISTRIBUTION
# ========================================
print("\n1. Creating histogram")
plt.figure(figsize=(9, 6))
plt.hist(df["Exam_Score"], bins=20, edgecolor="white")
plt.axvline(
    df["Exam_Score"].mean(),
    linestyle="--",
    label=f"Mean: {df['Exam_Score'].mean():.1f}%",
)
plt.title("Distribution of Exam Scores")
plt.xlabel("Exam score (%)")
plt.ylabel("Number of students")
plt.legend()
save_chart("01_exam_score_histogram.png")
plt.show()

# ========================================
# 2. SCATTER PLOT — STUDY HOURS VS SCORE
# ========================================
print("\n2. Creating scatter plot")
plt.figure(figsize=(9, 6))
plt.scatter(df["Hours_Studied"], df["Exam_Score"], alpha=0.55)
trend = np.poly1d(np.polyfit(df["Hours_Studied"], df["Exam_Score"], 1))
sorted_hours = np.sort(df["Hours_Studied"].unique())
plt.plot(sorted_hours, trend(sorted_hours), linestyle="--", label="Trend line")
plt.title("Study Hours and Exam Performance")
plt.xlabel("Hours studied")
plt.ylabel("Exam score (%)")
plt.legend()
save_chart("02_study_hours_scatter.png")
plt.show()

# ========================================
# 3. HORIZONTAL BAR — COURSE PERFORMANCE
# ========================================
print("\n3. Creating horizontal bar chart")
course_summary = (
    df.groupby("Course_Type")["Exam_Score"].mean().sort_values(ascending=True)
)
plt.figure(figsize=(9, 6))
bars = plt.barh(course_summary.index, course_summary.values)
plt.title("Average Exam Score by Course")
plt.xlabel("Average exam score (%)")
plt.ylabel("Course")
plt.xlim(0, 100)
for bar, value in zip(bars, course_summary.values):
    plt.text(value + 0.6, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center")
save_chart("03_course_horizontal_bar.png")
plt.show()

# ========================================
# 4. BOX PLOT — CODING EXPERIENCE
# ========================================
print("\n4. Creating box plot")
experience_order = ["Beginner", "Intermediate", "Advanced"]
experience_scores = [
    df.loc[df["Coding_Experience"] == level, "Exam_Score"]
    for level in experience_order
]
plt.figure(figsize=(9, 6))
plt.boxplot(experience_scores, tick_labels=experience_order, showmeans=True)
plt.title("Exam Score Spread by Coding Experience")
plt.xlabel("Coding experience")
plt.ylabel("Exam score (%)")
save_chart("04_experience_boxplot.png")
plt.show()

# ========================================
# 5. PIE CHART — PERFORMANCE LEVELS
# ========================================
print("\n5. Creating pie chart")
performance_order = ["Needs Support", "Satisfactory", "High Performer"]
performance_counts = (
    df["Performance_Level"].value_counts().reindex(performance_order).fillna(0)
)
plt.figure(figsize=(8, 8))
plt.pie(
    performance_counts.values,
    labels=performance_counts.index,
    autopct="%1.1f%%",
    startangle=90,
)
plt.title("Student Performance Categories")
save_chart("05_performance_pie.png")
plt.show()

# ========================================
# 6. LINE CHART — STUDY-HOUR BANDS
# ========================================
print("\n6. Creating line chart")
study_bins = [0, 2, 4, 6, 8, 10, 12.1]
study_labels = ["1-2", "2-4", "4-6", "6-8", "8-10", "10-12"]
df["Study_Hour_Band"] = pd.cut(
    df["Hours_Studied"], bins=study_bins, labels=study_labels, include_lowest=True
)
study_band_avg = df.groupby("Study_Hour_Band", observed=True)["Exam_Score"].mean()
plt.figure(figsize=(9, 6))
plt.plot(study_band_avg.index.astype(str), study_band_avg.values, marker="o")
plt.title("Average Score Across Study-Hour Bands")
plt.xlabel("Study-hour band")
plt.ylabel("Average exam score (%)")
plt.ylim(0, 100)
save_chart("06_study_band_line.png")
plt.show()

# ========================================
# 7. STACKED BAR — PASS/FAIL BY ATTENDANCE
# ========================================
print("\n7. Creating stacked bar chart")
attendance_order = ["Below 70%", "70-79%", "80-89%", "90%+"]
attendance_result = pd.crosstab(df["Attendance_Group"], df["Pass_Status"])
attendance_result = attendance_result.reindex(attendance_order).fillna(0)
attendance_percent = attendance_result.div(attendance_result.sum(axis=1), axis=0) * 100
attendance_percent = attendance_percent.reindex(columns=["Pass", "Fail"], fill_value=0)
plt.figure(figsize=(10, 6))
plt.bar(attendance_percent.index, attendance_percent["Pass"], label="Pass")
plt.bar(
    attendance_percent.index,
    attendance_percent["Fail"],
    bottom=attendance_percent["Pass"],
    label="Fail",
)
plt.title("Pass and Fail Shares by Attendance Group")
plt.xlabel("Attendance group")
plt.ylabel("Students (%)")
plt.ylim(0, 100)
plt.legend()
save_chart("07_attendance_stacked_bar.png")
plt.show()

# ========================================
# 8. HEATMAP — NUMERIC CORRELATIONS
# ========================================
print("\n8. Creating correlation heatmap")
numeric_columns = [
    "Hours_Studied",
    "Attendance",
    "Previous_GPA",
    "Sleep_Hours",
    "Projects_Completed",
    "Exam_Score",
]
corr = df[numeric_columns].corr()
plt.figure(figsize=(10, 8))
image = plt.imshow(corr, vmin=-1, vmax=1, aspect="auto")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.index)), corr.index)
for row in range(len(corr.index)):
    for col in range(len(corr.columns)):
        plt.text(col, row, f"{corr.iloc[row, col]:.2f}", ha="center", va="center")
plt.title("Correlation Matrix")
plt.colorbar(image, label="Correlation")
save_chart("08_correlation_heatmap.png")
plt.show()

# ========================================
# 9. LOLLIPOP CHART — PROJECT GROUPS
# ========================================
print("\n9. Creating lollipop chart")
project_order = ["0-3", "4-7", "8-11", "12-14"]
project_avg = (
    df.groupby("Project_Group", observed=True)["Exam_Score"]
    .mean()
    .reindex(project_order)
    .dropna()
)
positions = np.arange(len(project_avg))
plt.figure(figsize=(9, 6))
plt.vlines(positions, 0, project_avg.values)
plt.scatter(positions, project_avg.values, s=90)
plt.xticks(positions, project_avg.index)
plt.title("Average Exam Score by Projects Completed")
plt.xlabel("Projects completed")
plt.ylabel("Average exam score (%)")
plt.ylim(0, 100)
save_chart("09_projects_lollipop.png")
plt.show()

# Save a simple list of generated visualizations.
index_path = OUTPUT_DIR / "visualization_index.txt"
with index_path.open("w", encoding="utf-8") as index_file:
    index_file.write("GENERATED STUDENT VISUALIZATIONS\n")
    index_file.write("=" * 45 + "\n")
    for number, chart_path in enumerate(chart_paths, start=1):
        index_file.write(f"{number}. {chart_path.name}\n")

print(f"\n🎉 Created {len(chart_paths)} visualizations.")
print(f"✅ Visualization index saved as {index_path.name}")