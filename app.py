import streamlit as st
import pandas as pd
import joblib

model = joblib.load("placement_model.pkl")

st.set_page_config(page_title="Student Placement Prediction", page_icon="🎓", layout="centered")

st.markdown(
"""
<h1 style='text-align:center;color:#1E88E5;'>
🎓 Student Placement Prediction
</h1>
<p style='text-align:center;'>
Predict whether a student is likely to be placed.
</p>
""",
unsafe_allow_html=True
)

cgpa = st.slider("CGPA",0.0,10.0,7.5)
internships = st.slider("Internships",0,10,1)
projects = st.slider("Projects",0,20,2)
workshops = st.slider("Workshops/Certifications",0,20,2)
aptitude = st.slider("Aptitude Test Score",0,100,80)
softskills = st.slider("Soft Skills Rating",0,100,85)
activities = st.slider("Extracurricular Activities",0,10,2)
training = st.selectbox("Placement Training",["No","Yes"])
ssc = st.slider("SSC Marks",0,100,80)
hsc = st.slider("HSC Marks",0,100,85)

training = 1 if training=="Yes" else 0

if st.button("Predict"):

    data = pd.DataFrame({
        "CGPA":[cgpa],
        "Internships":[internships],
        "Projects":[projects],
        "Workshops/Certifications":[workshops],
        "AptitudeTestScore":[aptitude],
        "SoftSkillsRating":[softskills],
        "ExtracurricularActivities":[activities],
        "PlacementTraining":[training],
        "SSC_Marks":[ssc],
        "HSC_Marks":[hsc]
    })

    prediction = model.predict(data)[0]
    
    probability = model.predict_proba(data)

    confidence = max(probability[0])*100

    if prediction == 1:
       st.success("✅ Student is Likely to be Placed")
    else:
       st.error("❌ Student is Likely to be Not Placed")

    st.info(f"Prediction Confidence : {confidence:.2f}%")

    st.progress(int(confidence))