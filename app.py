import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

@st.cache_resource
def load_pretrained_model():
    model_path = "spemetryxyz.h5"
    model = load_model(model_path)
    return model

@st.cache_data
def load_csv_data():
    csv_path = "spemetryxyz.csv"
    df = pd.read_csv(csv_path)
    if "Student ID" in df.columns:
        df = df.drop("Student ID", axis=1)
    return df

model = load_pretrained_model()
spemetryxyz_df = load_csv_data()

csv_file_path = "xspemetryxyz.csv"

features = [
    "Official Grade", "Baseline RR", "Baseline RA", "Baseline RC",
    "Initial Grade", "Severity", "Intervention Approach", "Intervention Intensity"
]

A = spemetryxyz_df[features]

approach_mapping = {'Phonics': 1, 'Multi-Sensory': 2, 'Whole-Word': 3}
A['Intervention Approach'] = A['Intervention Approach'].map(approach_mapping)

severity_mapping = {'Very Mild': 1, 'Mild': 2, 'Moderate': 3, 'Severe': 4, 'Very Severe': 5}
A['Severity'] = A['Severity'].map(severity_mapping)

scaler = StandardScaler()
A_scaled = scaler.fit_transform(A)
pca = PCA(n_components=8)
A_pca = pca.fit_transform(A_scaled)

def map_to_numeric_grade(baseline_rr):
    if 53 <= baseline_rr <= 111:
        return 1
    elif 89 <= baseline_rr <= 149:
        return 2
    elif 107 <= baseline_rr <= 162:
        return 3
    elif 123 <= baseline_rr <= 180:
        return 4
    elif 139 <= baseline_rr <= 194:
        return 5
    else:
        return None

def determine_severity(official_grade, initial_grade):
    if official_grade is not None and initial_grade is not None:
        difference = official_grade - initial_grade
        if difference == 1:
            return 'Very Mild'
        elif difference == 2:
            return 'Mild'
        elif difference == 3:
            return 'Moderate'
        elif difference == 4:
            return 'Severe'
        elif difference >= 5:
            return 'Very Severe'
    return None

def save_student_data_to_csv(student_data, csv_file_path):
    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "Student ID", "Official Grade", "Baseline RR", "Baseline RA", "Baseline RC",
                "Initial Grade", "Severity", "Intervention Approach", "Intervention Intensity",
                "W01RR", "W01RA", "W01RC", "W02RR", "W02RA", "W02RC", "W03RR", "W03RA", "W03RC"
            ])
        if file_exists:
            with open(csv_file_path, 'r') as f:
                reader = csv.reader(f)
                count = sum(1 for row in reader)
            student_id = count  
        else:
            student_id = 1
        writer.writerow([
            student_id, student_data["Official Grade"], student_data["Baseline RR"],
            student_data["Baseline RA"], student_data["Baseline RC"], student_data["Initial Grade"],
            student_data["Severity"], student_data["Intervention Approach"], student_data["Intervention Intensity"],
            student_data["W01RR"], student_data["W01RA"], student_data["W01RC"],
            student_data["W02RR"], student_data["W02RA"], student_data["W02RC"],
            student_data["W03RR"], student_data["W03RA"], student_data["W03RC"]
        ])

def prepare_input(data, scaler, pca):
    data_scaled = scaler.transform(data)
    data_pca = pca.transform(data_scaled)
    return data_pca.reshape((-1, 1, data_pca.shape[1]))

def main():
    st.title("spemetry.xyz")
    st.write("Compare The Performance versus Potential of Your Dyslexic Student.")

    with st.form(key="student_form"):
        st.subheader("Student Form")
        official_grade = st.number_input("Official Grade (2-12)", min_value=0, value=1, step=1)
        baseline_rr = st.number_input("Baseline Reading Rate (words per minute)", min_value=0, value=50, step=1)
        baseline_ra = st.number_input("Baseline Reading Accuracy (correct words per minute)", min_value=0, value=50, step=1)
        baseline_rc = st.number_input("Baseline Reading Comprehension (comprehended correct words per minute)", min_value=0, value=50, step=1)
        
        intervention_approach = st.selectbox("Intervention Approach", 
                                             options=["Phonics", "Multi-Sensory", "Whole-Word"])
        intervention_intensity = st.number_input("Intervention Intensity (intervention hours per day)", min_value=0, value=1, step=1)
        
        st.subheader("First Three Weeks In Intervention")
        col1, col2, col3 = st.columns(3)
        with col1:
            w01_rr = st.number_input("Week 01 Reading Rate", min_value=0, value=50, step=1)
            w02_rr = st.number_input("Week 02 Reading Rate", min_value=0, value=50, step=1)
            w03_rr = st.number_input("Week 03 Reading Rate", min_value=0, value=50, step=1)
        with col2:
            w01_ra = st.number_input("Week 01 Reading Accuracy", min_value=0, value=50, step=1)
            w02_ra = st.number_input("Week 02 Reading Accuracy", min_value=0, value=50, step=1)
            w03_ra = st.number_input("Week 03 Reading Accuracy", min_value=0, value=50, step=1)
        with col3:
            w01_rc = st.number_input("Week 01 Reading Comprehension", min_value=0, value=50, step=1)
            w02_rc = st.number_input("Week 02 Reading Comprehension", min_value=0, value=50, step=1)
            w03_rc = st.number_input("Week 03 Reading Comprehension", min_value=0, value=50, step=1)
            
        submit_button = st.form_submit_button(label="Forecast")

    if submit_button:
        initial_grade = map_to_numeric_grade(baseline_rr)
        st.write(f"Initial Grade (Actual Grade): {initial_grade}")
        severity = determine_severity(official_grade, initial_grade)
        st.write(f"Severity: {severity}")

        student = {
            "Official Grade": official_grade,
            "Baseline RR": baseline_rr,
            "Baseline RA": baseline_ra,
            "Baseline RC": baseline_rc,
            "Initial Grade": initial_grade,
            "Severity": severity,
            "Intervention Approach": intervention_approach,
            "Intervention Intensity": intervention_intensity,
            "W01RR": w01_rr,
            "W01RA": w01_ra,
            "W01RC": w01_rc,
            "W02RR": w02_rr,
            "W02RA": w02_ra,
            "W02RC": w02_rc,
            "W03RR": w03_rr,
            "W03RA": w03_ra,
            "W03RC": w03_rc
        }
        
        save_student_data_to_csv(student, csv_file_path)
        st.success("The student data recorded successfully!")

        input_df = pd.DataFrame([student])
        X = input_df[features].copy()
        X['Intervention Approach'] = X['Intervention Approach'].map(approach_mapping)
        X['Severity'] = X['Severity'].map(severity_mapping)
        
        X_prepared = prepare_input(X, scaler, pca)
        predictions_A = model.predict(X_prepared)
        predictions_A = np.round(predictions_A)

        actual_values = pd.DataFrame([student], columns=[
            "W01RR", "W01RA", "W01RC", "W02RR", "W02RA", "W02RC", "W03RR", "W03RA", "W03RC"
        ])
        predictions_A_flat = predictions_A[0].flatten() 
        predicted_values = pd.DataFrame([predictions_A_flat], columns=[
            "W01RR", "W01RA", "W01RC", "W02RR", "W02RA", "W02RC", "W03RR", "W03RA", "W03RC"
        ])
        
        st.subheader("Comparison of The Actual vs Forecasted Values of The Student")
        st.write("In this graph, the actual values ​​show the student's performance, while the forecasted values ​​show his/her potential. If the actual values ​​are very close, equal or even above the forecasted values, the interventions you have implemented are sufficient, but if the actual values ​​are significantly below the forecasted values, you should reconsider the interventions you have implemented.")
        
        weeks = ['W01', 'W02', 'W03']
        rr_weeks = ['W01RR', 'W02RR', 'W03RR']
        ra_weeks = ['W01RA', 'W02RA', 'W03RA']
        rc_weeks = ['W01RC', 'W02RC', 'W03RC']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(weeks, actual_values[rr_weeks].values.flatten(), marker='o', linestyle='-', label='Actual RR')
        ax.plot(weeks, predicted_values[rr_weeks].values.flatten(), marker='x', linestyle='--', label='Forecasted RR')
        ax.plot(weeks, actual_values[ra_weeks].values.flatten(), marker='o', linestyle='-', label='Actual RA')
        ax.plot(weeks, predicted_values[ra_weeks].values.flatten(), marker='x', linestyle='--', label='Forecasted RA')
        ax.plot(weeks, actual_values[rc_weeks].values.flatten(), marker='o', linestyle='-', label='Actual RC')
        ax.plot(weeks, predicted_values[rc_weeks].values.flatten(), marker='x', linestyle='--', label='Forecasted RC')
        ax.set_xlabel('Weeks')
        ax.set_ylabel('Values')
        ax.set_title('Actual vs Forecasted Values for RR, RA, RC In First Three Weeks')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

if __name__ == '__main__':
    main()