"""
Compare three churn models on the Telco dataset using the SAME preprocessing.

Usage (from the project root):
    python compare_models.py data/raw/<your-csv-filename>.csv

Outputs: model_comparison.csv, confusion_matrices.png, roc_curves.png
Paste the printed table into Review 2, Section 9, and the images as screenshots.
"""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, ConfusionMatrixDisplay,
                             RocCurveDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv(sys.argv[1])

# Cleaning (same fix as your project): TotalCharges is text with blanks for tenure 0.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
y = (df["Churn"] == "Yes").astype(int)
X = df.drop(columns=["Churn", "customerID"], errors="ignore")

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

# One preprocessing step shared by every model, so differences come from the model only.
pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

# Stratify keeps the ~27% churn ratio equal in train and test.
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# class_weight="balanced" makes mistakes on the rare churn class cost more.
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

rows, fitted = [], {}
for name, clf in models.items():
    pipe = Pipeline([("pre", pre), ("model", clf)]).fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    proba = pipe.predict_proba(X_te)[:, 1]  # ROC-AUC needs probabilities, not labels
    fitted[name] = pipe
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_te, pred),
        "Precision (churn)": precision_score(y_te, pred),
        "Recall (churn)": recall_score(y_te, pred),
        "F1 (churn)": f1_score(y_te, pred),
        "ROC-AUC": roc_auc_score(y_te, proba),
    })

results = pd.DataFrame(rows).round(3)
results.to_csv("model_comparison.csv", index=False)
print(results.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, pipe) in zip(axes, fitted.items()):
    ConfusionMatrixDisplay.from_estimator(pipe, X_te, y_te, ax=ax, colorbar=False)
    ax.set_title(name)
plt.tight_layout(); plt.savefig("confusion_matrices.png", dpi=150)

fig, ax = plt.subplots(figsize=(6, 5))
for name, pipe in fitted.items():
    RocCurveDisplay.from_estimator(pipe, X_te, y_te, ax=ax, name=name)
ax.set_title("ROC curves")
plt.savefig("roc_curves.png", dpi=150)