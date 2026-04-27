# Water Quality Predictor
### HYDROSENSE-AI
https://hydrosense-ai.streamlit.app/

A machine learning project that predicts water quality based on physical, chemical, and biological parameters, with rule-based safety constraints.

## Problem Statement
To determine whether water is safe or unsafe for consumption using environmental data.

## Dataset
The dataset is based on real-world water quality data collected from Indian rivers and sources such as CPCB and government databases.

## Approach
- Data preprocessing and cleaning
- Feature selection (pH, DO, BOD, COD, Nitrate, Coliform, etc.)
- Label creation using standard safety ranges
- Model training using Random Forest Classifier
- In hindsight, while it may seem like there is no 'AI' used in this- the project actually uses about 60% of AI- the random forest model, pattern-based predictions- all core concepts of machine learning; the remaining 40% is purely rule-based (like thresholds, coliform override, etc). It's a Hybrid Intelligence System (HIS).

## Model
A Random Forest model is used to classify water into:
- Safe
- Moderate
- Unsafe

based on the data entered by the user.

## Features Used
- pH
- Dissolved Oxygen (DO)
- Biological Oxygen Demand (BOD)
- Chemical Oxygen Demand (COD)
- Nitrate
- Coliform

## Future Improvements
- Improve accuracy with larger datasets

## 
Built as a school project with a focus on practical AI applications. 
