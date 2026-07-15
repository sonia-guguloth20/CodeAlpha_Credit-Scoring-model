import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix

# =====================================================================
# STEP 1: LOAD & PREPROCESS MEDICAL DATA (UCI Wisconsin Breast Cancer)
# =====================================================================
print("Loading Breast Cancer Wisconsin dataset from scikit-learn...")
raw_data = load_breast_cancer()

# Convert to Pandas DataFrame for cleaner inspection and handling
X = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
y = pd.Series(raw_data.target) # 0 = Malignant (Cancerous), 1 = Benign (Non-cancerous)

# Align with typical medical risk target formats (1 = High Risk/Malignant, 0 = Healthy/Benign)
y = 1 - y 

print(f"Dataset loaded: {X.shape[0]} patient records, {X.shape[1]} biological features.")
print(f"Class distribution: {sum(y == 0)} Benign cases, {sum(y == 1)} Malignant cases.")

# =====================================================================
# STEP 2: SPLIT AND SCALE THE DATA
# =====================================================================
# Split data: 80% Training, 20% Testing (Stratified to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features (highly recommended for medical datasets)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================================
# STEP 3: TRAIN THE CLASSIFIER (Random Forest)
# =====================================================================
print("\nTraining Random Forest Classifier on patient data...")
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6)
model.fit(X_train_scaled, y_train)

# Generate Predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# =====================================================================
# STEP 4: EVALUATE MODEL PERFORMANCE
# =====================================================================
accuracy = model.score(X_test_scaled, y_test)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n" + "="*45)
print("             MEDICAL MODEL REPORT            ")
print("="*45)
print(f"Overall Accuracy: {accuracy * 100:.2f}%")
print(f"ROC-AUC Score:    {roc_auc:.4f}")
print("="*45)
print("\nClassification Report (Focus on Recall for high-risk diagnoses):")
print(classification_report(y_test, y_pred, target_names=["Benign (0)", "Malignant (1)"]))

# =====================================================================
# STEP 5: VISUALIZE AND EXPORT PLOTS
# =====================================================================
print("Generating and saving evaluation graphics...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=axes[0],
            xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
axes[0].set_title("Confusion Matrix")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

# Plot 2: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color='crimson', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('False Positive Rate (FPR)')
axes[1].set_ylabel('True Positive Rate (TPR / Sensitivity)')
axes[1].set_title('Receiver Operating Characteristic (ROC) Curve')
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.savefig("medical_model_performance.png")
print("Saved evaluation plot as 'medical_model_performance.png'")
plt.show()