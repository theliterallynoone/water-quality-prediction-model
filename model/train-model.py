import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

df= pd.read_csv("../data/water-quality.csv")
print("Dataset Preview:")
print(df.head())

drop_cols = ["Station", "City", "River", "State"]
for col in drop_cols:
    if col in df.columns:
        df = df.drop(columns=[col])
df = df.dropna()

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
print("\nClass Distribution:")
print(df["Quality"].value_counts())

features = ["pH", "DO", "BOD", "COD", "Nitrate", "Coliform"]

df = df.rename(columns={"Ph": "pH", "PH": "pH"})

X = df[features]
y = df["Quality"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model= RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))


# Example: Delhi Yamuna-like polluted water
sample = np.array([[6.5, 2.2, 12.0, 40, 3.5, 50000]])
prediction = model.predict(sample)
predicted_label = le.inverse_transform(prediction)
print("\nSample Prediction:", predicted_label[0])

#feature importance
importances = model.feature_importances_
feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(feature_importance)