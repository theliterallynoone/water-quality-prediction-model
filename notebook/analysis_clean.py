import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def classify_water(row: pd.Series) -> str:
    score = 0
    if row["DO"] < 4:
        score += 1
    if row["BOD"] > 6:
        score += 1
    if row["COD"] > 20:
        score += 1
    if row["Nitrate"] > 1:
        score += 1
    if row["Coliform"] > 500:
        score += 1

    if score >= 3:
        return "Unsafe"
    if score >= 1:
        return "Moderate"
    return "Safe"


# Load CSV using a path relative to this file (works no matter where you run from)
csv_path = Path(__file__).resolve().parent.parent / "data" / "water-quality.csv"
df = pd.read_csv(csv_path)
print(df.head())
print(df.describe())

df.hist(figsize=(10, 8))
plt.show()

plt.scatter(df['DO'], df['BOD'])
plt.xlabel('Dissolved Oxygen')
plt.ylabel('Biological Oxygen Demand')
plt.title('DO vs BOD relationship')
plt.show()

corr = df.corr(numeric_only=True)
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title('Correlation Matrix')
plt.show()

df['Quality']=df.apply(classify_water, axis=1)
df['Quality'].value_counts().plot(kind='bar')
plt.title('WaterQuality Distribution')
plt.show()