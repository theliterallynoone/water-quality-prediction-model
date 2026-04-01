import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df= pd.read_csv("data/water-quality.csv")
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

model= RandomForestClassifer(n_estimators=100)
model.fit(X, y_encoded)

#NOW I GET TO BUILD THE UI YAYYYYYYY
st.title("Water Quality Predictor") #add the shiny emoji thingy later
st.write("Enter water parameters to check if the water is safe")
pH= st.slider("pH", 0.0, 14.0, 7.0)
DO= st.slider("Dissolved Oxygen", 0.0, 10.0, 5.0)
BOD= st.slider("Biological Oxygen Demand", 0.0, 20.0, 3.0)
COD= st.slider("Chemical Oxygen Demand", 0.0, 50.0, 10.0)
nitrate= st.slider("Nitrate", 0.0, 10.0, 1.0)
coliform= st.slider("Coliform", 0.0, 50000, 100)

if st.button("Predict"):
    sample=np.array([[pH, DO, BOD, COD, nitrate, coliform]])
    prediction=model.predict(sample)
    result=le.inverse_transform(prediction)

    if result=='safe':
        st.success("The water is safe for consumption")
    elif result=='moderate':
        st.warning("The water is moderate for consumption/moderately safe")
    else:
        st.error("The water is unsafe for consumption")