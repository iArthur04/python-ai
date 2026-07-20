"""
DAY 2: Extended Real Data Analysis Project
Analyzing a realistic synthetic dataset of 500 students/learners.
"""

from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent
RANDOM_SEED = 42
N_STUDENTS = 500
PASS_MARK = 60

print("=" * 70)
print("📊 EXTENDED STUDENT PERFORMANCE ANALYSIS")
print("=" * 70)

# ========================================
# 1. CREATE A LARGER, MORE REALISTIC DATASET
# ========================================

print("\n🎓 1. CREATING STUDENT DATA")
rng = np.random.default_rng(RANDOM_SEED)

coding_levels = np.array(["Beginner", "Intermediate", "Advanced"])
course_types = np.array(["Web Dev", "AI/ML", "Data Science", "Cloud"])

hours_studied = rng.uniform(1, 12, N_STUDENTS).round(1)
attendance = np.clip(rng.normal(82, 10, N_STUDENTS), 55, 100).round(1)
previous_gpa = np.clip(rng.normal(3.0, 0.5, N_STUDENTS), 2.0, 4.0).round(2)
sleep_hours = np.clip(rng.normal(6.8, 1.1, N_STUDENTS), 4, 9).round(1)
coding_experience = rng.choice(coding_levels, N_STUDENTS, p=[0.45, 0.38, 0.17])
course_type = rng.choice(course_types, N_STUDENTS, p=[0.27, 0.25, 0.26, 0.22])
projects_completed = np.clip(rng.poisson(6, N_STUDENTS), 0, 14)

# Small effects for categorical variables.
experience_bonus = pd.Series(coding_experience).map(
    {"Beginner": 0.0, "Intermediate": 3.0, "Advanced": 5.5}
).to_numpy()
course_effect = pd.Series(course_type).map(
    {"Web Dev": 1.0, "AI/ML": -1.0, "Data Science": 0.5, "Cloud": 0.0}
).to_numpy()

# A balanced sleep pattern: performance is strongest near 7 hours.
sleep_effect = -1.15 * (sleep_hours - 7.0) ** 2 + 2.0

# Expected score combines several realistic performance drivers.
expected_score = (
    12
    + hours_studied * 3.8
    + attendance * 0.19
    + previous_gpa * 5.2
    + projects_completed * 0.65
    + sleep_effect
    + experience_bonus
    + course_effect
)

random_noise = rng.normal(0, 8.5, N_STUDENTS)
exam_score = np.clip(expected_score + random_noise, 0, 100).round(1)

student_data = {
    "Student_ID": range(1, N_STUDENTS + 1),
    "Hours_Studied": hours_studied,
    "Attendance": attendance,
    "Previous_GPA": previous_gpa,
    "Sleep_Hours": sleep_hours,
    "Coding_Experience": coding_experience,
    "Course_Type": course_type,
    "Projects_Completed": projects_completed,
    "Exam_Score": exam_score,
    "Expected_Score": np.clip(expected_score, 0, 100).round(1),
}

df = pd.DataFrame(student_data)
df["Performance_Gap"] = (df["Exam_Score"] - df["Expected_Score"]).round(1)
df["Pass_Status"] = np.where(df["Exam_Score"] >= PASS_MARK, "Pass", "Fail")
df["Performance_Level"] = pd.cut(
    df["Exam_Score"],
    bins=[-0.1, 59.9, 79.9, 100],
    labels=["Needs Support", "Satisfactory", "High Performer"],
)

print(f"📋 Dataset created: {len(df)} students and {len(df.columns)} features")
print(df.head())

# ========================================
# 2. EXPLORATORY DATA ANALYSIS
# ========================================

print("\n🔍 2. EXPLORATORY DATA ANALYSIS")
print("\nBasic statistics:")
print(df.select_dtypes(include=[np.number]).describe().round(2))

print("\n📊 Data types:")
print(df.dtypes)

print("\n🔢 Missing values:")
print(df.isna().sum())

print("\n🔢 Unique categorical values:")
print(f"Coding Experience: {df['Coding_Experience'].unique()}")
print(f"Course Type: {df['Course_Type'].unique()}")

# ========================================
# 3. ORIGINAL ANALYSIS QUESTIONS
# ========================================

print("\n❓ 3. ANSWERING ANALYSIS QUESTIONS")

# Q1: Average score and pass rate.
avg_score = df["Exam_Score"].mean()
pass_rate = (df["Pass_Status"] == "Pass").mean() * 100
print(f"Q1: Average exam score: {avg_score:.1f}%")
print(f"    Overall pass rate: {pass_rate:.1f}%")

# Q2: Course performance.
course_scores = df.groupby("Course_Type", observed=True)["Exam_Score"].mean().sort_values(ascending=False)
print("\nQ2: Average scores by course:")
for course, score in course_scores.items():
    print(f"  {course}: {score:.1f}%")

# Q3: Sleep and performance.
sleep_labels = ["Below 6", "6 to <7", "7 to <8", "8+"]
df["Sleep_Group"] = pd.cut(
    df["Sleep_Hours"], bins=[0, 6, 7, 8, 10], labels=sleep_labels, right=False
)
sleep_summary = df.groupby("Sleep_Group", observed=True)["Exam_Score"].agg(["mean", "count"])
print("\nQ3: Sleep hours versus exam scores:")
for group, row in sleep_summary.iterrows():
    print(f"  {group} hours: {row['mean']:.1f}% ({int(row['count'])} students)")

# Q4: Correlation with exam score.
numeric_factors = [
    "Hours_Studied",
    "Attendance",
    "Previous_GPA",
    "Sleep_Hours",
    "Projects_Completed",
    "Exam_Score",
]
correlations = df[numeric_factors].corr()
exam_correlations = correlations["Exam_Score"].drop("Exam_Score").sort_values(ascending=False)
print("\nQ4: Correlation with exam score:")
print(exam_correlations.round(3))

# ========================================
# 4. THREE ADDITIONAL INSIGHTS
# ========================================

print("\n💡 4. THREE ADDITIONAL INSIGHTS")

# Insight 1: Attendance groups and pass rates.
attendance_labels = ["Below 70%", "70-79%", "80-89%", "90%+"]
df["Attendance_Group"] = pd.cut(
    df["Attendance"], bins=[0, 70, 80, 90, 101], labels=attendance_labels, right=False
)
attendance_insight = df.groupby("Attendance_Group", observed=True).agg(
    Students=("Student_ID", "count"),
    Average_Score=("Exam_Score", "mean"),
    Pass_Rate=("Pass_Status", lambda values: (values == "Pass").mean() * 100),
)
print("\nInsight 1 — Attendance matters:")
print(attendance_insight.round(1))

# Insight 2: Practical projects and performance.
project_labels = ["0-3", "4-7", "8-11", "12-14"]
df["Project_Group"] = pd.cut(
    df["Projects_Completed"], bins=[-1, 3, 7, 11, 14], labels=project_labels
)
project_insight = df.groupby("Project_Group", observed=True).agg(
    Students=("Student_ID", "count"),
    Average_Score=("Exam_Score", "mean"),
    Pass_Rate=("Pass_Status", lambda values: (values == "Pass").mean() * 100),
)
print("\nInsight 2 — Completing more projects is linked with stronger results:")
print(project_insight.round(1))

# Insight 3: Identify hard-working students who perform well below expectations.
high_effort_threshold = df["Hours_Studied"].quantile(0.75)
underperformance_threshold = -10
high_effort_low_score = df[
    (df["Hours_Studied"] >= high_effort_threshold)
    & (df["Performance_Gap"] <= underperformance_threshold)
].copy()
print("\nInsight 3 — High-effort students needing targeted support:")
print(f"  High-effort threshold: {high_effort_threshold:.1f} study hours")
print(
    f"  Students studying at or above this level but scoring at least "
    f"{abs(underperformance_threshold)} points below expectation: {len(high_effort_low_score)}"
)
if not high_effort_low_score.empty:
    print(
        high_effort_low_score[
            [
                "Student_ID",
                "Hours_Studied",
                "Attendance",
                "Previous_GPA",
                "Sleep_Hours",
                "Projects_Completed",
                "Expected_Score",
                "Exam_Score",
                "Performance_Gap",
            ]
        ]
        .sort_values("Performance_Gap")
        .head(10)
        .to_string(index=False)
    )
else:
    print("  No students matched this risk pattern in the generated sample.")

# ========================================
# 5. TOP, OVER- AND UNDER-PERFORMERS
# ========================================

print("\n🏆 5. STUDENT-LEVEL FINDINGS")
top_students = df.nlargest(10, "Exam_Score")
print("\nTop 10 students:")
print(
    top_students[
        ["Student_ID", "Hours_Studied", "Attendance", "Projects_Completed", "Exam_Score"]
    ].to_string(index=False)
)

overperformers = df.nlargest(10, "Performance_Gap")
underperformers = df.nsmallest(10, "Performance_Gap")

print("\n🌟 Largest positive performance gaps:")
print(
    overperformers[
        ["Student_ID", "Expected_Score", "Exam_Score", "Performance_Gap"]
    ].to_string(index=False)
)

print("\n⚠️ Largest negative performance gaps:")
print(
    underperformers[
        ["Student_ID", "Expected_Score", "Exam_Score", "Performance_Gap"]
    ].to_string(index=False)
)

# ========================================
# 6. EXPORT RESULTS
# ========================================

print("\n💾 6. SAVING RESULTS")

csv_path = OUTPUT_DIR / "student_data_500.csv"
report_path = OUTPUT_DIR / "analysis_report_500.txt"
insights_path = OUTPUT_DIR / "student_insights_summary.csv"
support_path = OUTPUT_DIR / "students_needing_support.csv"

df.to_csv(csv_path, index=False)

summary_tables = []
for insight_name, table in [
    ("Attendance", attendance_insight),
    ("Projects", project_insight),
]:
    exported = table.reset_index()
    exported.insert(0, "Insight", insight_name)
    exported.rename(columns={exported.columns[1]: "Group"}, inplace=True)
    summary_tables.append(exported)
pd.concat(summary_tables, ignore_index=True).to_csv(insights_path, index=False)

high_effort_low_score.to_csv(support_path, index=False)

best_course = course_scores.index[0]
best_course_score = course_scores.iloc[0]
best_attendance_group = attendance_insight["Pass_Rate"].idxmax()
best_project_group = project_insight["Average_Score"].idxmax()

with report_path.open("w", encoding="utf-8") as report:
    report.write("=" * 70 + "\n")
    report.write("EXTENDED STUDENT PERFORMANCE ANALYSIS REPORT\n")
    report.write("=" * 70 + "\n\n")
    report.write(f"Total students: {len(df)}\n")
    report.write(f"Average exam score: {avg_score:.1f}%\n")
    report.write(f"Overall pass rate: {pass_rate:.1f}%\n")
    report.write(f"Best-performing course: {best_course} ({best_course_score:.1f}%)\n\n")

    report.write("THREE ADDITIONAL INSIGHTS\n")
    report.write("-" * 70 + "\n")
    report.write(
        f"1. Attendance: {best_attendance_group} had the highest pass rate "
        f"({attendance_insight.loc[best_attendance_group, 'Pass_Rate']:.1f}%).\n"
    )
    report.write(
        f"2. Projects: students completing {best_project_group} projects had the highest "
        f"average score ({project_insight.loc[best_project_group, 'Average_Score']:.1f}%).\n"
    )
    report.write(
        f"3. Targeted support: {len(high_effort_low_score)} high-effort students studied "
        f"at least {high_effort_threshold:.1f} hours but scored at least "
        f"{abs(underperformance_threshold)} points below their expected score.\n\n"
    )

    report.write("CORRELATIONS WITH EXAM SCORE\n")
    report.write("-" * 70 + "\n")
    for factor, value in exam_correlations.items():
        report.write(f"{factor}: {value:.3f}\n")

print(f"✅ Student data saved to: {csv_path.name}")
print(f"✅ Analysis report saved to: {report_path.name}")
print(f"✅ Insight summary saved to: {insights_path.name}")
print(f"✅ Support list saved to: {support_path.name}")