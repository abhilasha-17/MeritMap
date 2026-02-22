import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# -----------------------------
# LOAD MODEL + SCALER
# -----------------------------

model = load("placement_model.joblib")
scaler = load("placement_scaler.joblib")

FEATURE_COLUMNS = [
    'cgpa','college_tier','internships_count','projects_count',
    'certifications_count','coding_skill_score','aptitude_score',
    'communication_skill_score','logical_reasoning_score',
    'hackathons_participated','github_repos','linkedin_connections',
    'mock_interview_score','attendance_percentage','backlogs',
    'extracurricular_score','leadership_score','volunteer_experience',
    'sleep_hours','study_hours_per_day','branch_Civil','branch_ECE',
    'branch_EEE','branch_IT','branch_Mechanical'
]

st.set_page_config(page_title="MeritMap", layout="wide")

st.title("🎓 MeritMap")
st.markdown("### AI-Based Placement Readiness Evaluation System")
st.markdown("---")

# -----------------------------
# INPUT SECTION (2 Columns)
# -----------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("📘 Academic & Technical")
    cgpa = st.slider("CGPA", 4.0, 10.0, 7.0)
    coding = st.slider("Coding Skill Score", 0, 100, 60)
    aptitude = st.slider("Aptitude Score", 0, 100, 60)
    logical = st.slider("Logical Reasoning Score", 0, 100, 60)
    internships = st.slider("Internships Count", 0, 8, 1)
    projects = st.slider("Projects Count", 0, 13, 2)
    certifications = st.slider("Certifications Count", 0, 11, 2)
    backlogs = st.slider("Backlogs", 0, 6, 0)

with col2:
    st.subheader("💼 Personal & Engagement")
    communication = st.slider("Communication Skill Score", 0, 100, 60)
    mock = st.slider("Mock Interview Score", 0, 100, 60)
    leadership = st.slider("Leadership Score", 0, 100, 40)
    extracurricular = st.slider("Extracurricular Score", 0, 100, 50)
    github = st.slider("GitHub Repositories", 0, 16, 3)
    linkedin = st.slider("LinkedIn Connections", 50, 1000, 200)
    attendance = st.slider("Attendance Percentage", 50, 100, 75)
    sleep = st.slider("Sleep Hours", 3.0, 10.0, 7.0)
    study = st.slider("Study Hours per Day", 0.5, 10.0, 3.0)

college_tier = st.selectbox("College Tier", [1, 2, 3])
volunteer = st.selectbox("Volunteer Experience", [0, 1])
hackathons = st.slider("Hackathons Participated", 0, 8, 1)
branch = st.selectbox("Branch", ["Civil", "ECE", "EEE", "IT", "Mechanical"])

st.markdown("---")

# -----------------------------
# EVALUATION
# -----------------------------

if st.button("🚀 Evaluate Readiness"):

    input_dict = {
        'cgpa': cgpa,
        'college_tier': college_tier,
        'internships_count': internships,
        'projects_count': projects,
        'certifications_count': certifications,
        'coding_skill_score': coding,
        'aptitude_score': aptitude,
        'communication_skill_score': communication,
        'logical_reasoning_score': logical,
        'hackathons_participated': hackathons,
        'github_repos': github,
        'linkedin_connections': linkedin,
        'mock_interview_score': mock,
        'attendance_percentage': attendance,
        'backlogs': backlogs,
        'extracurricular_score': extracurricular,
        'leadership_score': leadership,
        'volunteer_experience': volunteer,
        'sleep_hours': sleep,
        'study_hours_per_day': study,
        'branch_Civil': 1 if branch == "Civil" else 0,
        'branch_ECE': 1 if branch == "ECE" else 0,
        'branch_EEE': 1 if branch == "EEE" else 0,
        'branch_IT': 1 if branch == "IT" else 0,
        'branch_Mechanical': 1 if branch == "Mechanical" else 0
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df[FEATURE_COLUMNS]

    scaled_input = scaler.transform(input_df)
    probability = model.predict_proba(scaled_input)[0][1]

    if probability < 0.20:
        category = "🔴 High Improvement Required"
        st.error(category)
    elif probability < 0.50:
        category = "🟡 Moderate Improvement Required"
        st.warning(category)
    elif probability < 0.80:
        category = "🟢 Strong Placement Readiness"
        st.success(category)
    else:
        category = "🔵 Excellent Placement Readiness"
        st.success(category)

    st.markdown("---")

    coef = model.coef_[0]
    contributions = scaled_input[0] * coef
    feature_contrib = dict(zip(FEATURE_COLUMNS, contributions))

    important_features = [
        'coding_skill_score','internships_count','mock_interview_score',
        'communication_skill_score','aptitude_score','cgpa','backlogs'
    ]

    contrib_filtered = {k: feature_contrib[k] for k in important_features}
    sorted_contrib = sorted(contrib_filtered.items(), key=lambda x: x[1])

    feedback_map = {
        'coding_skill_score': "Improve coding skills through consistent practice.",
        'internships_count': "Gain more internship experience.",
        'mock_interview_score': "Practice mock interviews.",
        'communication_skill_score': "Work on communication skills.",
        'aptitude_score': "Strengthen aptitude preparation.",
        'cgpa': "Focus on improving academic consistency.",
        'backlogs': "Clearing backlogs will significantly improve readiness."
    }

    strengths_map = {
        'coding_skill_score': "Strong coding foundation.",
        'internships_count': "Good internship exposure.",
        'mock_interview_score': "Strong interview performance.",
        'communication_skill_score': "Good communication ability.",
        'aptitude_score': "Strong aptitude performance.",
        'cgpa': "Solid academic record.",
        'backlogs': "Minimal academic backlogs."
    }

    weak_areas = [f for f, v in sorted_contrib if v < 0][:3]
    strong_areas = sorted(contrib_filtered.items(), key=lambda x: x[1], reverse=True)
    strong_areas = [f for f, v in strong_areas if v > 0][:2]

    colA, colB = st.columns(2)

    with colA:
        st.subheader("⚠ Areas to Improve")
        if weak_areas:
            for f in weak_areas:
                st.write("•", feedback_map[f])
        else:
            st.write("No major weak areas detected.")

    with colB:
        st.subheader("💪 Strengths")
        if strong_areas:
            for f in strong_areas:
                st.write("•", strengths_map[f])
        else:
            st.write("Balanced profile.")

    st.markdown("---")
    st.caption("MeritMap | made by @Abhilasha")