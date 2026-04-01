from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Water Quality App", layout="wide",)
page=st.sidebar.selectbox("Navigate", ["Home", "About"])

_ROOT = Path(__file__).resolve().parent
df = pd.read_csv(_ROOT / "data" / "water-quality.csv")
df=df.dropna()

def classify_water(row):
    score=0
    if row['Coliform'] > 500:
        score += 1
    if row['DO'] < 4:
        score += 1
    if row['BOD'] > 6:
        score += 1
    if row['COD'] > 20:
        score += 1
    if row['Nitrate'] > 1:
        score += 1
    if row['pH'] < 6.5 or row['pH'] > 8.5:
        score += 1
    if score >= 3:
        return "Unsafe"
    elif score >= 1:
        return "Moderate"
    else:
        return "Safe"

def explain_result(pH, DO, BOD, COD, Nitrate, Coliform):
    reasons = []
    if Coliform > 500:
        reasons.append("High coliform levels indicate possible sewage contamination")
    if DO < 4:
        reasons.append("Low dissolved oxygen harms aquatic life")
    if BOD > 6:
        reasons.append("High BOD means high organic pollution")
    if COD > 20:
        reasons.append("High COD indicates chemical pollution")
    if Nitrate > 1:
        reasons.append("Elevated nitrate levels may be due to fertilizers or waste")
    if pH < 6.5 or pH > 8.5:
        reasons.append("pH outside safe range")
    return reasons
    
df["Quality"] = df.apply(classify_water, axis=1)
features = ["pH", "DO", "BOD", "COD", "Nitrate", "Coliform"]

x=df[features]
y=df["Quality"]

le=LabelEncoder()
y_encoded=le.fit_transform(y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x, y_encoded)

#NOW I GET TO BUILD THE UI YAYYYYYYY
if page == "Home":
    st.title("Water Quality Predictor") #add the shiny emoji thingy later
    st.write("Enter water parameters to check if the water is safe")
    pH= st.slider("pH", 0.0, 14.0, 7.0)
    DO= st.slider("Dissolved Oxygen", 0.0, 10.0, 5.0)
    BOD= st.slider("Biological Oxygen Demand", 0.0, 20.0, 3.0)
    COD= st.slider("Chemical Oxygen Demand", 0.0, 50.0, 10.0)
    Nitrate= st.slider("Nitrate", 0.0, 10.0, 1.0)
    coliform = st.slider("Coliform", 0.0, 50000.0, 100.0, step=1.0)

    if st.button("Predict"):
        sample = pd.DataFrame([[pH, DO, BOD, COD, Nitrate, coliform]], columns=features)
        prediction = model.predict(sample)
        label = le.inverse_transform(prediction)[0]

        if label == "Safe":
            st.success("The water is safe for consumption")
        elif label == "Moderate":
            st.warning("The water is moderate for consumption/moderately safe") #i gotta pick out a better statement
        else:
            st.error("The water is unsafe for consumption")

        reasons = explain_result(pH, DO, BOD, COD, Nitrate, coliform)
        if reasons:
            st.subheader("Why this result?") #gotta add a magnifying glass emoji later
            for r in reasons:
                st.write("- " + r)

elif page == "About":
    st.header("About This Project")
    st.subheader("Project Overview")
    st.write("""This project uses machine learning to predict water quality based on environmental parameters such as pH, dissolved oxygen, and pollution indicators. 
    It analyzes the data and classifies water as Safe, Moderate, or Unsafe.""")

    st.subheader("Here's How It Works") 
    st.write("""
    - Input water parameters
    - Compare with standard safety ranges
    - Apply machine learning model (Random Forest)
    - Generate prediction + explanation
    """)

    st.subheader("🌍 Sustainable Development Goals")
    st.write("""
    - SDG 6: Clean Water and Sanitation 💧
    - SDG 3: Good Health and Well-being 🏥
    - SDG 13: Climate Action 🌱
    """)

    st.subheader("Development")
    st.write("""
    Built as a school project using Python, Machine Learning, and Streamlit.
    """)

st.markdown("---")
st.caption("Built using Machine Learning • Streamlit App • 2026")