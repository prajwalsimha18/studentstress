import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------- DISPLAY SETTINGS ----------------

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# ---------------- LOAD DATASET ----------------

df = pd.read_csv("student_data.csv")

print("\n================ DATASET LOADED SUCCESSFULLY ================\n")
print(df.head())

# ---------------- PREPROCESSING ----------------

print("\n================ PREPROCESSING STARTED ================\n")

# Remove unnecessary columns
df = df.drop(columns=["student_id"])

# Remove missing values
df = df.dropna()

# Encode categorical columns
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

print("Preprocessing Completed Successfully")

# ---------------- FEATURES & TARGET ----------------

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# ---------------- TRAIN TEST SPLIT ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n================ TRAIN TEST SPLIT COMPLETED ================\n")

print(f"Training Data Shape: {X_train.shape}")
print(f"Testing Data Shape: {X_test.shape}")

# ---------------- MODEL GENERATION ----------------

print("\n================ MODEL GENERATION ================\n")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

print("Random Forest Model Trained Successfully")

# ---------------- MODEL EVALUATION ----------------

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n================ MODEL EVALUATION ================\n")

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R2 Score: {r2:.2f}")

# ---------------- FEATURE IMPORTANCE ----------------

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

print("\n================ FEATURE IMPORTANCE ================\n")
print(importance_df.head(10))

# ---------------- CUSTOM PREDICTION ----------------

print("\n================ CUSTOM PREDICTION ================\n")

age = int(input("Enter Age: "))
study_hours = float(input("Enter Study Hours Per Day: "))
social_media = float(input("Enter Social Media Hours: "))
attendance = float(input("Enter Attendance Percentage: "))
sleep = float(input("Enter Sleep Hours: "))
previous_gpa = float(input("Enter Previous GPA (0-4): "))
stress = float(input("Enter Stress Level (1-10): "))
screen_time = float(input("Enter Screen Time: "))
motivation = int(input("Enter Motivation Level (1-10): "))
time_management = float(input("Enter Time Management Score: "))

# ---------------- CREATE CUSTOM INPUT ----------------

custom_input = pd.DataFrame([{
    'age': age,
    'gender': 1,
    'major': 3,
    'study_hours_per_day': study_hours,
    'social_media_hours': social_media,
    'netflix_hours': 1.0,
    'part_time_job': 0,
    'attendance_percentage': attendance,
    'sleep_hours': sleep,
    'diet_quality': 1,
    'exercise_frequency': 3,
    'parental_education_level': 2,
    'internet_quality': 1,
    'mental_health_rating': 7,
    'extracurricular_participation': 1,
    'previous_gpa': previous_gpa,
    'semester': 4,
    'stress_level': stress,
    'dropout_risk': 0,
    'social_activity': 3,
    'screen_time': screen_time,
    'study_environment': 2,
    'access_to_tutoring': 1,
    'family_income_range': 1,
    'parental_support_level': 7,
    'motivation_level': motivation,
    'exam_anxiety_score': 6,
    'learning_style': 2,
    'time_management_score': time_management
}])

# ---------------- PREDICT SCORE ----------------

predicted_score = model.predict(custom_input)

# Manual adjustments for realism
if attendance < 50:
    predicted_score[0] -= 20

if study_hours < 3:
    predicted_score[0] -= 10

if stress > 8:
    predicted_score[0] -= 10

if social_media > 8:
    predicted_score[0] -= 5

if sleep < 5:
    predicted_score[0] -= 5

# Keep prediction within range
predicted_score[0] = max(0, min(100, predicted_score[0]))

# ---------------- FINAL RESULT ----------------

print("\n================ PREDICTION RESULT ================\n")
print(f"Predicted Exam Score: {predicted_score[0]:.2f}")

print("\n================ PROJECT COMPLETED SUCCESSFULLY ================\n")
print("AIML Mini Project Demo Successful")