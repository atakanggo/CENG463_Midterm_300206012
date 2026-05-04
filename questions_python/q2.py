import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score, 
                             average_precision_score, matthews_corrcoef, balanced_accuracy_score,
                             brier_score_loss, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. Dataset Generation (Extreme Imbalance)
# ==========================================
# Simulating Credit Card Fraud: 100,000 samples, ~0.4% positive class
X, y = make_classification(n_samples=100000, n_features=20, n_informative=10, 
                           n_redundant=5, weights=[0.996], flip_y=0, random_state=42)

pos_count = sum(y == 1)
neg_count = sum(y == 0)
imbalance_ratio = neg_count / pos_count

print(f"Dataset Shape: {X.shape}")
print(f"Positive Class (Fraud): {pos_count} ({(pos_count / len(y)) * 100:.2f}%)")
print(f"Negative Class (Normal): {neg_count}")
print(f"Imbalance Ratio (IR): {imbalance_ratio:.2f}:1\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ==========================================
# 2. Models & Cost-Sensitive Setup
# ==========================================
scale_pos_weight = neg_count / pos_count

models = {
    'Logistic Regression (Baseline)': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest (Cost-Sensitive)': RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
    'XGBoost (Cost-Sensitive)': XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss'),
    'MLP Neural Net': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
}

# ==========================================
# 3. Resampling Strategies
# ==========================================
samplers = {
    'None': None,
    'SMOTE': SMOTE(random_state=42),
    'ADASYN': ADASYN(random_state=42),
    'RUS': RandomUnderSampler(random_state=42)
}

def evaluate_model(y_true, y_pred, y_prob):
    return {
        'Recall': recall_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'PR-AUC': average_precision_score(y_true, y_prob),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }

results = []
print("Training Models with Resampling Strategies...\n")
for model_name, model in models.items():
    for sampler_name, sampler in samplers.items():
        steps = [('scaler', StandardScaler())]
        if sampler is not None:
            steps.append(('sampler', sampler))
        steps.append(('model', model))
        
        pipeline = ImbPipeline(steps)
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        metrics = evaluate_model(y_test, y_pred, y_prob)
        metrics['Model'] = model_name
        metrics['Resampling'] = sampler_name
        results.append(metrics)

results_df = pd.DataFrame(results).set_index(['Model', 'Resampling'])
print("--- Baseline vs Cost-Sensitive vs Resampling Results ---")
print(results_df.to_string())

# ==========================================
# 4. Probability Calibration
# ==========================================
print("\n--- Performing Probability Calibration (XGBoost) ---")
# 4a. Train standard uncalibrated model for comparison
xgb_base = ImbPipeline([('scaler', StandardScaler()), ('model', XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42))])
xgb_base.fit(X_train, y_train)
prob_uncalibrated = xgb_base.predict_proba(X_test)[:, 1]

# 4b. Train calibrated model using internal Cross-Validation (cv=3 fixes the error)
calibrated_xgb = CalibratedClassifierCV(estimator=xgb_base, method='isotonic', cv=3)
calibrated_xgb.fit(X_train, y_train)
prob_calibrated = calibrated_xgb.predict_proba(X_test)[:, 1]

print(f"Brier Score (Uncalibrated): {brier_score_loss(y_test, prob_uncalibrated):.5f}")
print(f"Brier Score (Calibrated): {brier_score_loss(y_test, prob_calibrated):.5f}")

# Reliability Diagram
fig, ax = plt.subplots(figsize=(8, 6))
CalibrationDisplay.from_predictions(y_test, prob_uncalibrated, n_bins=10, name="XGBoost (Uncalibrated)", ax=ax)
CalibrationDisplay.from_predictions(y_test, prob_calibrated, n_bins=10, name="XGBoost (Isotonic Calibration)", ax=ax)
plt.title("Reliability Diagram (Calibration Curve)")
plt.show()

# ==========================================
# 5. Precision-Recall Trade-off & Optimal Threshold
# ==========================================
precisions, recalls, thresholds = precision_recall_curve(y_test, prob_calibrated)
f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8) 
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"\nOptimal Threshold (Max F1): {optimal_threshold:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, label='PR Curve', color='blue')
plt.scatter(recalls[optimal_idx], precisions[optimal_idx], color='red', marker='o', s=100, label=f'Optimal F1 (Thresh={optimal_threshold:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve with Optimal Threshold')
plt.legend()
plt.show()

# ==========================================
# 6. Confusion Matrix at Optimal Threshold
# ==========================================
y_pred_optimal = (prob_calibrated >= optimal_threshold).astype(int)

cm = confusion_matrix(y_test, y_pred_optimal)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Fraud'])
disp.plot(cmap='Blues', values_format='d')
plt.title(f"Confusion Matrix (Threshold = {optimal_threshold:.2f})")
plt.show()