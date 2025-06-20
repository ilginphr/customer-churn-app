# 📊 Customer Churn Prediction App

This project is a **Machine Learning web application** built to predict the likelihood of a customer leaving a service (churning), based on demographic and usage-related inputs.

It includes two parts:
- 🎯 **Model training** with Python and scikit-learn
- 🌐 **Web application** using Streamlit

---

## 💡 Project Goal

The primary goal is to:
- Use historical customer data to train a classification model
- Allow users to input customer information through a user-friendly interface
- Predict whether the customer is likely to churn or stay
- Provide the **churn probability** as feedback

---

## 📁 Files & Structure

| File / Folder        | Description                                      |
|----------------------|--------------------------------------------------|
| `churn_model.py`     | ML model training script (Logistic Regression)   |
| `churn_model.pkl`    | Trained model file (saved with `joblib`)         |
| `churn_data.csv`     | Dataset used to train the model                  |
| `churn_app.py`       | Streamlit app that collects input and predicts   |
| `requirements.txt`   | List of all required Python packages             |
| `venv/`              | Virtual environment folder (ignored in GitHub)   |

---

## 🧠 Machine Learning Workflow

1. **Data Preprocessing**
   - Loaded a customer dataset with features like gender, tenure, contract type, etc.
   - Applied **OneHotEncoding** for categorical variables (e.g., gender, internet service).
   - Split the data into training and test sets.

2. **Model Training**
   - Chose **Logistic Regression** (simple and interpretable for classification).
   - Trained the model to classify `Churn` vs `No Churn`.
   - Saved the model using `joblib` for deployment.

3. **Web Application**
   - Built with **Streamlit**.
   - User inputs are collected with sliders, dropdowns, and number fields.
   - The input is encoded exactly like during training.
   - The model predicts `churn` or `no churn` and returns a probability.

---

## 🚀 How to Run the App

```bash
git clone https://github.com/ilginphr/customer-churn-app
cd customer-churn-app
pip install -r requirements.txt
streamlit run churn_app.py
```

---

## 🎓 Educational Value

This project is **perfect for beginners** learning Machine Learning because it teaches:

- 🧹 Data cleaning and feature engineering (OneHotEncoding, normalization)
- 📊 Model training and saving
- 🧪 Model evaluation and prediction
- 🖥️ Web deployment using **Streamlit**
- 🧠 Understanding of classification problems and probabilities

By building this from scratch, I learned not just how to train a model, but also how to **turn it into a real product** with user interaction.

---

## 🔮 Sample Prediction

After inputting customer data (like contract type, tenure, monthly charges), the app will return:

> ✅ This customer is **likely to stay**. Estimated probability: %89.4  
> or  
> ❌ This customer is **likely to churn**. Estimated probability: %76.2

---

## 📌 Dependencies

- Python ≥ 3.10  
- `pandas`  
- `scikit-learn`  
- `joblib`  
- `streamlit`  

(See `requirements.txt` for full list)