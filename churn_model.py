import pandas as pd
import joblib

# 1. csv dosyasini pands dataframe i olarak yukledik
df = pd.read_csv("churn_data.csv")

#print("🔎 İlk 5 satır:")
#print(df.head())

#print("\n📊 Veri Bilgisi:")
#print(df.info())

#print("\n📈 İstatistiksel Özet:")
#print(df.describe())

#print("\n❓ Eksik Değerler:")
#print(df.isnull().sum())


#print("\n🧠 Kategorik Değişkenler:")
#categoricals = df.select_dtypes(include="object").columns
#print(categoricals)

#print("\n🔢 Sayısal Değişkenler:")
#numerics = df.select_dtypes(include="number").columns
#print(numerics)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# customerID model için anlamlı değil-butun sutunu siler
df = df.drop("customerID", axis=1)
#axis=1 her sutun icin
#axis=0 her satir icin

# sayiya cevir object gibi okunmus
#errors="coerce" → sayı olmayanları NaN yapar.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# imputation-Oluşturdugumuz NaN değerleri ortalama yap
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())

# onehotencoding- true => multicollinearity engeller
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("Churn_Yes", axis=1) #features-input
y = df_encoded["Churn_Yes"] #labels-output

# 7. train-0.8 test-0.2
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=2000) # % degeri verir
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
#Genel başarı oranı
print("Precision:", precision_score(y_test, y_pred))
#“Ayrılacak” dediklerinin doğruluğu
print("Recall:", recall_score(y_test, y_pred))
#Gerçek ayrılanları yakalama oranı
print("F1 Score:", f1_score(y_test, y_pred))
#Precision & Recall dengesi

joblib.dump(model, "churn_model.pkl") #Modeli .pkl dosyasına yazar (save)
print("✅ Model başarıyla churn_model.pkl dosyasına kaydedildi.")
print(model)