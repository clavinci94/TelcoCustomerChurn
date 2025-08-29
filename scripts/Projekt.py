# ============================
# 1. Setup & Libraries
# ============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ============================
# 2. Daten laden
# ============================
data = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")


print("Shape:", data.shape)
print(data.head())

# ============================
# 3. Erste Analyse
# ============================
print(data.info())
print(data['Churn'].value_counts())

sns.countplot(x='Churn', data=data)
plt.title("Kundenabwanderung (0 = Nein, 1 = Ja)")
plt.show()

# ============================
# 4. Datenvorbereitung
# ============================
data = data.drop("customerID", axis=1)

# Konvertiere 'TotalCharges' zu numerisch
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data = data.dropna()

# Kategorische Features encoden
for col in data.select_dtypes(include=['object']).columns:
    if col != 'Churn':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

# Zielvariable
y = data['Churn'].map({'Yes': 1, 'No': 0})
X = data.drop('Churn', axis=1)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================
# 5. Modelltraining
# ============================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ============================
# 6. Ergebnisse
# ============================
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
import os

# Ordner "results" erstellen, falls er noch nicht existiert
os.makedirs("results", exist_ok=True)

# Plot speichern
import os
from sklearn.metrics import confusion_matrix

# Confusion Matrix berechnen
cm = confusion_matrix(y_test, y_pred)

# Ordner "results" erstellen, falls er noch nicht existiert
os.makedirs("results", exist_ok=True)

# Plot speichern
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join("results", "confusion_matrix.png"))
plt.show()

# Feature Importance (Random Forest)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

importances = rf_model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=importance_df)
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.savefig(os.path.join("results", "feature_importance.png"))
plt.show()