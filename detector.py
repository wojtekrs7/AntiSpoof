import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import math


def make_model(model_name: str = "svm") -> Pipeline:
    if model_name == "svm":
        base = SVC(
            C=2.0,
            kernel="rbf",
            gamma="scale",
            probability=True,
        )
    elif model_name == "logreg":
        base = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        )
    elif model_name == "rf":
        base = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
    else:
        raise ValueError(f"Nieznany model: {model_name!r}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", base),
    ])


def _eer(fpr, tpr):
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fnr[i] + fpr[i]) / 2)


def train_model(X: np.ndarray, y: np.ndarray, model_name: str = "svm"):
    Xtr, Xval, ytr, yval = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    clf = make_model(model_name)
    clf.fit(Xtr, ytr)

    pred_tr = clf.predict(Xtr)
    acc_tr = balanced_accuracy_score(ytr, pred_tr)

    if hasattr(clf, "predict_proba"):
        proba_val = clf.predict_proba(Xval)[:, 1]  # prawdopodobienstwo SPOOF
        pred_val = (proba_val >= 0.5).astype(int)
        acc_val = balanced_accuracy_score(yval, pred_val)
        auc_val = roc_auc_score(yval, proba_val)
        fpr, tpr, _ = roc_curve(yval, proba_val)
        eer_val = _eer(fpr, tpr)

        print(f"[{model_name}] Train  balanced acc: {acc_tr:.3f}")
        print(f"[{model_name}] Valid  balanced acc: {acc_val:.3f} | "
              f"AUC: {auc_val:.3f} | EER: {eer_val:.3f}")
    else:
        pred_val = clf.predict(Xval)
        acc_val = balanced_accuracy_score(yval, pred_val)
        print(f"[{model_name}] Train  balanced acc: {acc_tr:.3f}")
        print(f"[{model_name}] Valid  balanced acc: {acc_val:.3f}")

    return clf

def train_svm(X: np.ndarray, y: np.ndarray):
    return train_model(X, y, model_name="svm")


def predict_file(model, feats_row: np.ndarray) -> int:
    return int(model.predict(feats_row)[0])
