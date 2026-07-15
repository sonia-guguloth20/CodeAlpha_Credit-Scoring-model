import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix

# Set random seed for reproducibility
np.random.seed(42)

# =====================================================================
# STEP 1: GENERATE SYNTHETIC FINANCIAL DATA (Makes submission self-contained!)
# =====================================================================
print("Generating synthetic financial history dataset...")
num_samples = 2000

data = {
    'income': np.random.normal(55000, 15000, num_samples).clip(15000, 150000),
    'debt': np.random.normal(20000, 12000, num_samples).clip(0, 80000),
    'payment_history_score': np.random.randint(30, 100, num_samples), # Higher = better history
    'utilization_rate': np.random.uniform(0.1, 0.95, num_samples),    # Credit card balance used / limit
    'age': np.random.randint(21, 65, num_samples)
}

df = pd.DataFrame(data)

# =====================================================================
# STEP 2: FEATURE ENGINEERING
# =====================================================================
print("Performing feature engineering...")
# 1. Debt-to-Income (DTI) Ratio: High DTI is a classic risk factor
df['debt_to_income'] = df['debt'] / df['income']

# 2. Risk Score (Ground truth logic): Determines if someone defaults
# Probability of default increases with high DTI, poor payment score, and high card utilization
default_probability = (
    0.4 * (df['debt_to_income']) +
    0.4 * (1 - (df['payment_history_score'] / 100)) +
    0.2 * df['utilization_rate']
)

# If probability exceeds threshold, assign as Default (1 = High Risk/Default, 0 = Creditworthy)
df['default'] = (default_probability > 0.45).astype(int)

# =====================================================================
# STEP 3: PREPARE DATA FOR MODELING
# =====================================================================
X = df.drop(columns=['default'])
y = df['default']

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features for stability
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================================
# STEP 4: TRAIN CLASSIFIER (Random Forest)
# =====================================================================
print("Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# =====================================================================
# STEP 5: ACCURACY & PERFORMANCE ASSESSMENT
# =====================================================================
accuracy = model.score(X_test_scaled, y_test)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n" + "="*45)
print("             MODEL METRICS REPORT            ")
print("="*45)
print(f"Overall Model Accuracy: {accuracy * 100:.2f}%")
print(f"ROC-AUC Score:          {roc_auc:.4f}")
print("="*45)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Creditworthy (0)", "Default/High Risk (1)"]))

# =====================================================================
# STEP 6: VISUALIZE RESULTS (Saves submission-ready plots)
# =====================================================================
print("Generating evaluation plots...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=["Creditworthy", "Default"], yticklabels=["Creditworthy", "Default"])
axes[0].set_title("Confusion Matrix")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

# Plot 2: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('Receiver Operating Characteristic (ROC) Curve')
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.savefig("credit_model_performance.png")
print("Saved model report plot to 'credit_model_performance.png'")
plt.show()