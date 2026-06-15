A Machine Learning project that predicts a student's exam score based on academic, lifestyle, and personal factors using a Random Forest Regressor model.

The project includes:

🖥️ Command Line Version (aiml_demo.py)
🌐 Interactive Streamlit Web App (streamlit_app.py)
🤖 Machine Learning Model using Random Forest Regression
📊 Student Performance Analysis

The application trains a model on student data and predicts exam scores using factors such as:

Study Hours
Attendance Percentage
Sleep Hours
Social Media Usage
Previous GPA
Stress Levels
Motivation Levels
Time Management Skills
And more

The project uses Streamlit, Pandas, and Scikit-Learn.

📂 Project Structure
Student-Exam-Score-Predictor/
│
├── student_data.csv
├── aiml_demo.py
├── streamlit_app.py
├── requirements.txt
└── README.md
🚀 Features
Machine Learning
Data preprocessing
Missing value handling
Label Encoding
Random Forest Regression
Model evaluation using:
MAE (Mean Absolute Error)
R² Score
Web Application
Modern UI
Real-time predictions
Interactive input forms
Dataset preview
Performance metrics dashboard
Prediction Factors

The model considers multiple factors including:

Academic performance
Study habits
Attendance
Sleep schedule
Mental health
Family support
Screen time
Motivation level
Learning style

The model implementation is based on a Random Forest pipeline with preprocessing and evaluation metrics.

🛠 Requirements

Install the required packages:

pip install -r requirements.txt

Dependencies include:

streamlit
pandas
scikit-learn

💻 Windows Setup
1. Clone Repository
git clone <YOUR_REPOSITORY_URL>
cd Student-Exam-Score-Predictor
2. Create Virtual Environment
python -m venv venv
3. Activate Virtual Environment
Command Prompt
venv\Scripts\activate
PowerShell
.\venv\Scripts\Activate.ps1
4. Install Dependencies
pip install -r requirements.txt
5. Run Streamlit Application
streamlit run streamlit_app.py

The application will open automatically in your browser.

🍎 macOS Setup
1. Clone Repository
git clone <YOUR_REPOSITORY_URL>
cd Student-Exam-Score-Predictor
2. Create Virtual Environment
python3 -m venv venv
3. Activate Virtual Environment
source venv/bin/activate
4. Install Dependencies
pip install -r requirements.txt

If pip points to Python 2:

pip3 install -r requirements.txt
5. Run Streamlit Application
streamlit run streamlit_app.py

If Streamlit is not detected:

python3 -m streamlit run streamlit_app.py
🖥 Running the Command Line Version

Run:

python aiml_demo.py

or

python3 aiml_demo.py

You will be prompted to enter:

Age
Study Hours
Attendance
Sleep Hours
Stress Level
GPA
Motivation Level
Screen Time
Time Management Score

The model will then predict the expected exam score.

📊 Model Information
Algorithm
Random Forest Regressor
Train/Test Split
80% Training
20% Testing
Evaluation Metrics
Mean Absolute Error (MAE)
R² Score

The model automatically performs categorical encoding and feature preprocessing before training.

🎯 Sample Prediction

Input:

Study Hours: 6
Attendance: 90
Sleep Hours: 8
Stress Level: 4
Previous GPA: 3.5

Output:

Predicted Exam Score: 84.27
🔮 Future Improvements
Model export using Pickle
XGBoost implementation
Student performance analytics dashboard
Study recommendations
PDF report generation
Cloud deployment
User authentication
👨‍💻 Author

Prajwal Simha MP

Machine Learning & AI Enthusiast
