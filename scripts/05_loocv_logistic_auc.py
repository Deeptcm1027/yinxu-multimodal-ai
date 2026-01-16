import os
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

IN_PATH = "data/example/master_features_example.csv"
OUT_DIR = "outputs"
OUT_CSV = os.path.join(OUT_DIR, "loocv_auc_results.csv")

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(IN_PATH)

features = ["enlarged_raw", "gut_pc1", "IFN_score"]
X = df[features].values
y = df["yin_bin"].values
ids = df["subject_id"].values

loo = LeaveOneOut()
probs, y_true, y_pred, used_ids = [], [], [], []

for tr, te in loo.split(X):
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y[tr], y[te]

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = LogisticRegression(solver="liblinear")
    model.fit(X_tr_s, y_tr)

    p = model.predict_proba(X_te_s)[0, 1]
    pred = int(p >= 0.5)

    probs.append(p)
    y_true.append(int(y_te[0]))
    y_pred.append(pred)
    used_ids.append(ids[te][0])

auc = roc_auc_score(y_true, probs)
acc = accuracy_score(y_true, y_pred)

pd.DataFrame({
    "subject_id": used_ids,
    "yin_bin_true": y_true,
    "prob_yin_high": probs,
    "pred": y_pred
}).to_csv(OUT_CSV, index=False)

print("LOOCV AUC:", round(auc, 3))
print("LOOCV Accuracy:", round(acc, 3))
print("Saved:", OUT_CSV)
