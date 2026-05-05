import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# -------------------------------
# LOAD DATA
# -------------------------------
print("\n--- LOADING DATA ---")
df = pd.read_csv("data/ds_salaries.csv")

print("\nFirst 5 rows:\n", df.head())
print("\nShape of dataset:", df.shape)

# -------------------------------
# DATA UNDERSTANDING
# -------------------------------
print("\n--- DATA INFO ---")
print("\nColumn Names:\n", df.columns)

print("\nDataset Info:")
df.info()

print("\nMissing Values:\n", df.isnull().sum())

# -------------------------------
# DATA CLEANING
# -------------------------------
print("\n--- CLEANING DATA ---")
df.drop_duplicates(inplace=True)
df.fillna("Unknown", inplace=True)

print("\nAfter Cleaning:\n", df.isnull().sum())

# -------------------------------
# EDA
# -------------------------------
print("\n--- EDA ---")

print("\nTop Job Roles:\n", df["job_title"].value_counts().head(10))
print("\nExperience Levels:\n", df["experience_level"].value_counts())
print("\nEmployment Types:\n", df["employment_type"].value_counts())

# -------------------------------
# VISUALIZATION (saved as images)
# -------------------------------
print("\n--- VISUALIZATION ---")

df.groupby("experience_level")["salary_in_usd"].mean().plot(kind="bar")
plt.title("Average Salary by Experience Level")
plt.savefig("salary_vs_experience.png")
plt.close()

top_jobs = df["job_title"].value_counts().head(10)
top_jobs.plot(kind="bar")
plt.title("Top 10 Job Roles")
plt.xticks(rotation=45)
plt.savefig("top_jobs.png")
plt.close()

# -------------------------------
# MODEL BUILDING
# -------------------------------
print("\n--- MODEL BUILDING ---")

# Drop unnecessary columns
df_model = df.drop(["salary", "salary_currency"], axis=1)

# Reduce job_title noise
top_jobs = df_model["job_title"].value_counts().head(10).index
df_model["job_title"] = df_model["job_title"].apply(
    lambda x: x if x in top_jobs else "Other"
)

# One-hot encoding
df_model = pd.get_dummies(df_model, drop_first=True)

# Features & Target
X = df_model.drop("salary_in_usd", axis=1)
y = df_model["salary_in_usd"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print("\nFinal Model R² Score:", score)

# -------------------------------
# FEATURE IMPORTANCE
# -------------------------------
importance = model.feature_importances_

feat_importance = pd.Series(importance, index=X.columns)
top_features = feat_importance.sort_values(ascending=False).head(10)

top_features.plot(kind="barh")
plt.title("Top 10 Important Features")
plt.savefig("feature_importance.png")
plt.close()

# -------------------------------
# EXPORT CLEAN DATA FOR POWER BI
# -------------------------------
df.to_csv("cleaned_data.csv", index=False)

print("\nCleaned dataset exported as 'cleaned_data.csv'")