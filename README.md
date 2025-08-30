# Telco Customer Churn Prediction

Projektbeschreibung
Ziel dieses Projekts ist es, die Abwanderung (Churn) von Kunden in einem Telekommunikationsunternehmen vorherzusagen.  
Dazu wurden Machine-Learning-Modelle (Logistic Regression, Random Forest, XGBoost) trainiert und verglichen.

## Tech Stack
- Python (pandas, numpy, scikit-learn, matplotlib, seaborn)
- Jupyter Notebook
- Dataset: [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Ergebnisse
- Bestes Modell: Random Forest mit Accuracy von 79%  
- Wichtigste Features: Vertragsdauer, Vertragsart, monatliche Kosten  
- Business-Empfehlung: Kunden mit Monatsverträgen haben höhere Kündigungsrate → Empfehlung: Jahresabos fördern.

## Ordnerstruktur
- `data/` → Dataset  
- `notebooks/` → Analyse & Visualisierungen  
- `scripts/` → Training Code  
- `results/` → Plots  

---
