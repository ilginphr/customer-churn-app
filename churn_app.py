import streamlit as st
import pandas as pd
import joblib  #deployement-ml urunlestirme

model = joblib.load("churn_model.pkl")
st.title("🔮 Customer Churn Prediction App")
st.write("This app predicts the likelihood of a customer leaving the service based on the information you provide below.")

gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Has a Partner?", ["Yes", "No"])
dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthlycharges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

input_dict = { #dictionary
    "gender": [gender],
    "SeniorCitizen": [senior],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "MonthlyCharges": [monthlycharges],
    "Contract": [contract],
    "InternetService": [internet]
}

#dictionary’i tek satırlık veri tablosuna (DataFrame) çeviriyor:
input_df = pd.DataFrame(input_dict)
# 6. OneHotEncoding 
input_encoded = pd.get_dummies(input_df)

# 7. Eğitimde olmayan sütunları da ekleyelim (eksik olanları sıfırla)
model_columns = model.feature_names_in_  # scikit-learn 1.0+ ile otomatik
for col in model_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

input_encoded = input_encoded[model_columns] 
#hem tüm sütunlara sahip, hem de doğru sırada olduğundan emin olur.

if st.button("Predict"):
    prediction = model.predict(input_encoded)[0] #model predict her zaman 0 ya da 1 verir(>0.5=1)
    probability = model.predict_proba(input_encoded)[0][1]

    if prediction == 1:
        st.error(f"❌This customer is **likely to churn**. Estimated probability: %{round(probability * 100, 2)}")
    else:
        st.success(f"✅ This customer is **likely to stay**. Estimated probability:  %{round((1 - probability) * 100, 2)}")
