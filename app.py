from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

_ROOT = Path(__file__).resolve().parent
df = pd.read_csv(_ROOT / "data" / "water-quality.csv")
df=df.dropna()

def classify_water(row):
    score=0
    if row['DO'] < 4:
        score += 1
    if row['BOD'] > 6:
        score += 1
    if row['COD'] > 20:
        score += 1
    if row['Nitrate'] > 1:
        score += 1
    if row['Coliform'] > 500:
        score += 1

    if score >= 3:
        return "Unsafe"
    elif score >= 1:
        return "Moderate"
    else:
        return "Safe"
    
df["Quality"] = df.apply(classify_water, axis=1)
features = ["pH", "DO", "BOD", "COD", "Nitrate", "Coliform"]

x=df[features]
y=df["Quality"]

le=LabelEncoder()
y_encoded=le.fit_transform(y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x, y_encoded)

#NOW I GET TO BUILD THE UI YAYYYYYYY
st.title("Water Quality Predictor") #add the shiny emoji thingy later
st.write("Enter water parameters to check if the water is safe")
pH= st.slider("pH", 0.0, 14.0, 7.0)
DO= st.slider("Dissolved Oxygen", 0.0, 10.0, 5.0)
BOD= st.slider("Biological Oxygen Demand", 0.0, 20.0, 3.0)
COD= st.slider("Chemical Oxygen Demand", 0.0, 50.0, 10.0)
nitrate= st.slider("Nitrate", 0.0, 10.0, 1.0)
coliform = st.slider("Coliform", 0.0, 50000.0, 100.0, step=1.0)

if st.button("Predict"):
    sample = pd.DataFrame([[pH, DO, BOD, COD, nitrate, coliform]], columns=features)
    prediction = model.predict(sample)
    label = le.inverse_transform(prediction)[0]

    if label == "Safe":
        st.success("The water is safe for consumption")
    elif label == "Moderate":
        st.warning("The water is moderate for consumption/moderately safe") #i gotta pick out a better statement
    else:
        st.error("The water is unsafe for consumption")