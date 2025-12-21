# -----------------------------------------------------------
# ❤ HEART DISEASE PREDICTION – FINAL TRAINING CODE (ML + DL + SHAP)
# -----------------------------------------------------------

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)

# Deep Learning Imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model

# SHAP
import shap
shap.initjs()

# -----------------------------------------------------------
# 📁 Create output folder
# -----------------------------------------------------------
os.makedirs("model", exist_ok=True)

# -----------------------------------------------------------
# 📂 Load Dataset
# -----------------------------------------------------------
df = pd.read_csv("heart.csv")

# Features and Target
X = df.drop("target", axis=1)
y = df["target"]

# -----------------------------------------------------------
# 📊 Correlation Matrix
# -----------------------------------------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Heart Disease Features')
plt.tight_layout()
plt.savefig("model/correlation_matrix.png")
plt.close()

# -----------------------------------------------------------
# ⚙ Scaling
# -----------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------------------------
# ✂ Train-Test Split (Stratified)
# -----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------------------------------
# 🎯 MACHINE LEARNING MODELS
# -----------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

print("\n🧠 Training ML Models...\n")
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    roc = roc_auc_score(y_test, preds)

    print(f"🔹 {name} | Accuracy: {acc:.3f} | ROC-AUC: {roc:.3f}")
    print(classification_report(y_test, preds))

    results.append((name, acc, roc))

# -----------------------------------------------------------
# ⭐ Best ML Model = Gradient Boosting
# -----------------------------------------------------------
best_model = models["Gradient Boosting"]

# -----------------------------------------------------------
# 📉 Confusion Matrix (Gradient Boosting)
# -----------------------------------------------------------
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Gradient Boosting")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("model/confusion_matrix.png")
plt.close()

# -----------------------------------------------------------
# 📈 ROC Curve (Gradient Boosting)
# -----------------------------------------------------------
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

plt.figure()
plt.plot(fpr, tpr, label='Gradient Boosting (AUC = {:.2f})'.format(
    roc_auc_score(y_test, y_pred_proba)))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig("model/roc_curve.png")
plt.close()


# -----------------------------------------------------------
# 🤖 DEEP LEARNING MODEL (ANN)
# -----------------------------------------------------------
ann = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

ann.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n🧠 Training ANN (Deep Learning Model)...\n")

history = ann.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=0
)

loss, ann_acc = ann.evaluate(X_test, y_test, verbose=0)

print(f"🔹 ANN Model | Accuracy: {ann_acc:.3f}\n")

results.append(("ANN (Deep Learning)", ann_acc, np.nan))


# -----------------------------------------------------------
# 🧠 SHAP EXPLAINER
# -----------------------------------------------------------
explainer = shap.Explainer(best_model)
shap_values = explainer(X_test)

# SHAP Bar Plot
plt.figure()
shap.plots.bar(shap_values, show=False)
plt.title("SHAP Feature Importance")
plt.savefig("model/shap_bar.png")
plt.close()

# SHAP Summary Plot
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("model/shap_summary.png")
plt.close()

# -----------------------------------------------------------
# 💾 Save Models
# -----------------------------------------------------------
pickle.dump(best_model, open("model/model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))
ann.save("model/ann_model.h5")

print("\n✅ Training Complete!")
print("📊 All graphs saved in 'model/' folder")
print("🤖 ANN model saved as ann_model.h5")