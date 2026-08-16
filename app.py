import streamlit as st
import pandas as pd
import joblib

model = joblib.load("LogisticRegression_heart.pkl")
expected_columns = joblib.load("columns.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Heart Stroke prediction by Meet")
st.markdown("Provide the following Details")

age = st.slider("Age", 18,100, 40)
sex = st.selectbox("Sex", ['Male', 'Female'])
chestPain = st.selectbox("Chest Pain Type" , ['ATA', 'NAP', 'TA', 'ASY'])
restingBP = st.number_input("Resting Blood Pressure (mm of hg)", 80,200,120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100,600,200)
fastingBS = st.selectbox("Fasting Blood Sugar", [0,1])
restingECG = st.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])
maxHR = st.slider("Max Heart Rate", 60, 220, 150)
ExerciseAngina = st.selectbox("Excercise-induced angina", ['Y', 'N'])
oldPeak =  st.slider("oldPeak (ST depression)", 0.0, 6.0, 1.0)
stSlope = st.selectbox("ST Slope", ['Up', 'Flat', 'Down'])


if st.button("Predict"):
    raw_input = {
        "Age" : age,
        "RestingBP" : restingBP,
        "Cholesterol" : cholesterol,
        "FastingBS" : fastingBS,
        "MaxHR" : maxHR,
        "OldPeak" : oldPeak,
        "Sex_" + sex : 1,
        "ChestPainType_" + chestPain : 1,
        "RestingECG_" + restingECG : 1,
        "ExerciseAngina_" + ExerciseAngina:1,
        "ST_Slope_" + stSlope:1 
    }

    input_df = pd.DataFrame([raw_input])

    for cols in expected_columns:
        if cols not in input_df.columns:
            input_df[cols] = 0

    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.erro("High Rsk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")