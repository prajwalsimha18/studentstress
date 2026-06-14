from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "student_data.csv"
TARGET_COLUMN = "exam_score"


st.set_page_config(
    page_title="Student Exam Score Predictor",
    layout="wide",
)


def load_styles():
    st.markdown(
        """
        <style>
        :root {
            --page: #f5f8f6;
            --surface: #ffffff;
            --surface-soft: #edf6f3;
            --ink: #17211f;
            --muted: #5f6f69;
            --line: #d8e4de;
            --teal: #0f766e;
            --teal-dark: #0b4f4a;
            --coral: #d35d4d;
            --gold: #c99122;
            --blue: #2f5f98;
        }

        .stApp {
            background:
                radial-gradient(circle at 8% 8%, rgba(15, 118, 110, 0.14), transparent 28%),
                radial-gradient(circle at 92% 2%, rgba(211, 93, 77, 0.12), transparent 25%),
                linear-gradient(180deg, #f5f8f6 0%, #eef6f3 46%, #f8faf9 100%);
            color: var(--ink);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stMainBlockContainer"] {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3, p, label, span {
            letter-spacing: 0 !important;
        }

        .app-hero {
            position: relative;
            overflow: hidden;
            padding: 2.1rem 2.25rem;
            border: 1px solid rgba(15, 118, 110, 0.18);
            border-radius: 8px;
            background:
                linear-gradient(135deg, rgba(15, 118, 110, 0.95), rgba(47, 95, 152, 0.9)),
                linear-gradient(90deg, #0f766e, #2f5f98);
            box-shadow: 0 22px 50px rgba(31, 68, 61, 0.16);
            margin-bottom: 1.2rem;
        }

        .app-hero::after {
            content: "";
            position: absolute;
            right: -4rem;
            top: -7rem;
            width: 18rem;
            height: 18rem;
            border: 1px solid rgba(255, 255, 255, 0.32);
            transform: rotate(24deg);
        }

        .hero-label {
            position: relative;
            z-index: 1;
            width: fit-content;
            padding: 0.28rem 0.65rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.16);
            color: #f7fffc !important;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
        }

        .hero-title {
            position: relative;
            z-index: 1;
            margin: 0.85rem 0 0.45rem;
            max-width: 780px;
            color: #ffffff !important;
            font-size: clamp(2.1rem, 5vw, 4.2rem);
            line-height: 1.02;
            font-weight: 800;
        }

        .hero-copy {
            position: relative;
            z-index: 1;
            max-width: 640px;
            margin: 0;
            color: rgba(255, 255, 255, 0.84) !important;
            font-size: 1.03rem;
        }

        div[data-testid="stMetric"] {
            padding: 1.15rem 1.25rem;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.86);
            box-shadow: 0 10px 28px rgba(36, 65, 57, 0.08);
        }

        [data-testid="stMetricLabel"] p {
            color: var(--muted) !important;
            font-weight: 700;
        }

        [data-testid="stMetricValue"] {
            color: var(--ink);
            font-weight: 800;
        }

        [data-testid="stForm"] {
            padding: 1.35rem 1.4rem 1.5rem;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.9);
            box-shadow: 0 16px 38px rgba(36, 65, 57, 0.09);
        }

        [data-testid="stForm"] h3 {
            color: var(--ink);
            font-size: 1.45rem;
        }

        [data-testid="stForm"] [data-testid="stWidgetLabel"] p,
        [data-testid="stForm"] label,
        [data-testid="stForm"] label p {
            color: var(--ink) !important;
            font-weight: 700;
        }

        .section-label {
            margin: 0.45rem 0 0.85rem;
            color: var(--teal-dark);
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
        }

        [data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #fbfdfc;
        }

        .stButton > button {
            min-height: 3rem;
            border: 0;
            border-radius: 8px;
            background: linear-gradient(135deg, var(--teal), var(--blue));
            color: white;
            font-weight: 800;
            box-shadow: 0 12px 24px rgba(15, 118, 110, 0.2);
        }

        .stButton > button:hover {
            border: 0;
            color: white;
            filter: brightness(1.04);
        }

        [data-testid="stForm"] [data-baseweb="input"] {
            border-radius: 8px;
            border: 1px solid #dbe7e2;
            background: #ffffff !important;
        }

        [data-testid="stForm"] [data-baseweb="base-input"],
        [data-testid="stForm"] input {
            background: #ffffff !important;
            color: var(--ink) !important;
            caret-color: var(--teal);
        }

        [data-testid="stForm"] button[kind="secondary"],
        [data-testid="stForm"] [data-testid="stNumberInput"] button {
            border-color: #dbe7e2 !important;
            background: #f5f8f6 !important;
            color: var(--teal-dark) !important;
        }

        [data-testid="stSlider"] [role="slider"] {
            border-color: var(--coral);
            background: var(--coral);
        }

        .result-card {
            margin-top: 1rem;
            padding: 1.3rem 1.45rem;
            border: 1px solid rgba(15, 118, 110, 0.2);
            border-left: 6px solid var(--teal);
            border-radius: 8px;
            background: linear-gradient(135deg, #ffffff, #edf6f3);
            box-shadow: 0 14px 34px rgba(36, 65, 57, 0.1);
        }

        .result-card.warning {
            border-left-color: var(--gold);
            background: linear-gradient(135deg, #ffffff, #fbf5e7);
        }

        .result-card.danger {
            border-left-color: var(--coral);
            background: linear-gradient(135deg, #ffffff, #fff0ed);
        }

        .result-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
        }

        .result-score {
            margin-top: 0.2rem;
            color: var(--ink);
            font-size: 2.6rem;
            line-height: 1;
            font-weight: 850;
        }

        .result-copy {
            margin-top: 0.75rem;
            color: var(--muted);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH).dropna()


@st.cache_resource
def train_model():
    df = load_data().drop(columns=["student_id"], errors="ignore")
    label_encoders = {}

    encoded_df = df.copy()
    for column in encoded_df.select_dtypes(include="object").columns:
        encoder = LabelEncoder()
        encoded_df[column] = encoder.fit_transform(encoded_df[column])
        label_encoders[column] = encoder

    X = encoded_df.drop(columns=[TARGET_COLUMN])
    y = encoded_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
        "rows": len(df),
        "features": X.columns.tolist(),
    }

    defaults = {}
    for column in X.columns:
        if column in label_encoders:
            defaults[column] = df[column].mode().iloc[0]
        else:
            defaults[column] = float(df[column].median())

    return model, label_encoders, metrics, defaults, df


def number_input(label, column, defaults, min_value, max_value, step):
    return st.number_input(
        label,
        min_value=min_value,
        max_value=max_value,
        value=float(defaults[column]),
        step=step,
    )


def select_input(label, column, defaults, label_encoders):
    options = label_encoders[column].classes_.tolist()
    default_index = options.index(defaults[column]) if defaults[column] in options else 0
    return st.selectbox(label, options, index=default_index)


def encode_input(raw_input, label_encoders, feature_order):
    encoded = {}
    for column in feature_order:
        value = raw_input[column]
        if column in label_encoders:
            encoded[column] = int(label_encoders[column].transform([value])[0])
        else:
            encoded[column] = value

    return pd.DataFrame([encoded], columns=feature_order)


def adjust_prediction(prediction, raw_input):
    adjusted = float(prediction)

    if raw_input["attendance_percentage"] < 50:
        adjusted -= 20
    if raw_input["study_hours_per_day"] < 3:
        adjusted -= 10
    if raw_input["stress_level"] > 8:
        adjusted -= 10
    if raw_input["social_media_hours"] > 8:
        adjusted -= 5
    if raw_input["sleep_hours"] < 5:
        adjusted -= 5

    return max(0, min(100, adjusted))


model, label_encoders, metrics, defaults, original_df = train_model()

load_styles()

st.markdown(
    """
    <section class="app-hero">
        <div class="hero-label">AIML Mini Project</div>
        <h1 class="hero-title">Student Exam Score Predictor</h1>
        <p class="hero-copy">Estimate student performance from study routine, attendance, lifestyle, and academic profile inputs.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
col1.metric("Dataset Rows", metrics["rows"])
col2.metric("Mean Absolute Error", f"{metrics['mae']:.2f}")
col3.metric("R2 Score", f"{metrics['r2']:.2f}")

with st.form("prediction_form"):
    st.subheader("Student Details")
    st.markdown('<div class="section-label">Core inputs</div>', unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        age = st.number_input("Age", min_value=15, max_value=35, value=int(defaults["age"]))
        study_hours = number_input("Study hours per day", "study_hours_per_day", defaults, 0.0, 12.0, 0.5)
        social_media = number_input("Social media hours", "social_media_hours", defaults, 0.0, 12.0, 0.5)
        attendance = number_input("Attendance percentage", "attendance_percentage", defaults, 0.0, 100.0, 1.0)
        previous_gpa = number_input("Previous GPA", "previous_gpa", defaults, 0.0, 4.0, 0.1)
    with right:
        sleep = number_input("Sleep hours", "sleep_hours", defaults, 0.0, 12.0, 0.5)
        stress = number_input("Stress level", "stress_level", defaults, 1.0, 10.0, 0.5)
        screen_time = number_input("Screen time", "screen_time", defaults, 0.0, 18.0, 0.5)
        motivation = st.slider("Motivation level", 1, 10, int(defaults["motivation_level"]))
        time_management = number_input("Time management score", "time_management_score", defaults, 0.0, 10.0, 0.5)

    with st.expander("More student profile fields"):
        st.markdown('<div class="section-label">Profile and environment</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            gender = select_input("Gender", "gender", defaults, label_encoders)
            major = select_input("Major", "major", defaults, label_encoders)
            part_time_job = select_input("Part-time job", "part_time_job", defaults, label_encoders)
            diet_quality = select_input("Diet quality", "diet_quality", defaults, label_encoders)
            internet_quality = select_input("Internet quality", "internet_quality", defaults, label_encoders)
            dropout_risk = select_input("Dropout risk", "dropout_risk", defaults, label_encoders)
            learning_style = select_input("Learning style", "learning_style", defaults, label_encoders)
        with c2:
            parental_education = select_input(
                "Parental education level",
                "parental_education_level",
                defaults,
                label_encoders,
            )
            study_environment = select_input("Study environment", "study_environment", defaults, label_encoders)
            tutoring = select_input("Access to tutoring", "access_to_tutoring", defaults, label_encoders)
            income = select_input("Family income range", "family_income_range", defaults, label_encoders)
            extracurricular = select_input(
                "Extracurricular participation",
                "extracurricular_participation",
                defaults,
                label_encoders,
            )

        c3, c4 = st.columns(2)
        with c3:
            netflix_hours = number_input("Netflix hours", "netflix_hours", defaults, 0.0, 12.0, 0.5)
            exercise = st.slider("Exercise frequency", 0, 7, int(defaults["exercise_frequency"]))
            semester = st.slider("Semester", 1, 8, int(defaults["semester"]))
            social_activity = st.slider("Social activity", 0, 10, int(defaults["social_activity"]))
        with c4:
            mental_health = number_input("Mental health rating", "mental_health_rating", defaults, 1.0, 10.0, 0.5)
            parental_support = st.slider("Parental support level", 1, 10, int(defaults["parental_support_level"]))
            exam_anxiety = st.slider("Exam anxiety score", 1, 10, int(defaults["exam_anxiety_score"]))

    submitted = st.form_submit_button("Predict Exam Score", width="stretch")

raw_input = {
    "age": age,
    "gender": gender,
    "major": major,
    "study_hours_per_day": study_hours,
    "social_media_hours": social_media,
    "netflix_hours": netflix_hours,
    "part_time_job": part_time_job,
    "attendance_percentage": attendance,
    "sleep_hours": sleep,
    "diet_quality": diet_quality,
    "exercise_frequency": exercise,
    "parental_education_level": parental_education,
    "internet_quality": internet_quality,
    "mental_health_rating": mental_health,
    "extracurricular_participation": extracurricular,
    "previous_gpa": previous_gpa,
    "semester": semester,
    "stress_level": stress,
    "dropout_risk": dropout_risk,
    "social_activity": social_activity,
    "screen_time": screen_time,
    "study_environment": study_environment,
    "access_to_tutoring": tutoring,
    "family_income_range": income,
    "parental_support_level": parental_support,
    "motivation_level": motivation,
    "exam_anxiety_score": exam_anxiety,
    "learning_style": learning_style,
    "time_management_score": time_management,
}

if submitted:
    model_input = encode_input(raw_input, label_encoders, metrics["features"])
    predicted_score = model.predict(model_input)[0]
    final_score = adjust_prediction(predicted_score, raw_input)

    if final_score >= 75:
        result_class = "result-card"
        result_message = "Strong predicted performance. Keep the routine consistent."
    elif final_score >= 50:
        result_class = "result-card warning"
        result_message = "Moderate predicted performance. Attendance, sleep, and study time can improve the score."
    else:
        result_class = "result-card danger"
        result_message = "Low predicted performance. Focus first on attendance, regular study time, and reducing stress."

    st.markdown(
        f"""
        <section class="{result_class}">
            <div class="result-label">Predicted exam score</div>
            <div class="result-score">{final_score:.2f} / 100</div>
            <div class="result-copy">{result_message}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.progress(int(round(final_score)))

with st.expander("Preview training data"):
    st.dataframe(original_df.head(20), width="stretch")
