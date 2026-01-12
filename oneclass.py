from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
import joblib

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import extract_features_from_wav


@dataclass
class OCMeta:
    sr: int = 16000
    feat: str = "mfcc"
    note: str = "One-class CM; score kalibrowany przez empiryczne CDF z próbek enroll."
    oc_model_name: str = "ocsvm"
    # przechowujemy rozkład wyników na próbkach enroll (do kalibracji)
    enroll_scores_sorted: list[float] | None = None


def make_oc_model(model_name: str = "ocsvm") -> Pipeline:

    if model_name == "ocsvm":
        base = OneClassSVM(
            kernel="rbf",
            gamma="scale",
            nu=0.1,        # możesz sobie potem dostroić
        )
    elif model_name == "iforest":
        base = IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=42,
        )
    else:
        raise ValueError(f"Nieznany model one-class: {model_name!r}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("oc", base),
    ])


def fit_enroll(feats_all: np.ndarray, model_name: str = "ocsvm"):
    model = make_oc_model(model_name)
    model.fit(feats_all)

    scores = model.decision_function(feats_all).astype(float)
    scores_sorted = np.sort(scores).tolist()

    meta = OCMeta(
        oc_model_name=model_name,
        enroll_scores_sorted=scores_sorted,
    )
    return model, meta



def save_enroll(model, meta, model_path):

    model_path = Path(model_path)
    payload = {
        "model": model,
        "meta": meta,
    }
    joblib.dump(payload, model_path)


def load_enroll(model_path):

    model_path = Path(model_path)
    obj = joblib.load(model_path)

    if isinstance(obj, dict) and "model" in obj and "meta" in obj:
        model = obj["model"]
        meta = obj["meta"]
        return model, meta

    raise ValueError(
        f"Plik {model_path} nie zawiera słownika {{'model','meta'}}. "
        "Przetrenuj model komendą `antispoof oc-enroll` (albo nową wersją `oc-train`).")

def score_to_percent_notspoof(score: float, enroll_scores_sorted: list[float]) -> float:
    arr = np.array(enroll_scores_sorted, dtype=float)
    k = np.searchsorted(arr, score, side="right")
    perc = 100.0 * k / max(1, len(arr))
    return float(np.clip(perc, 0.0, 100.0))

def predict_percent_notspoof(model, meta: OCMeta, wav_path: Path) -> tuple[float, float]:
    feats = extract_features_from_wav(
        Path(wav_path),
        feature_types=getattr(meta, "feat", "mfcc"),
    ).reshape(1, -1)

    s = float(model.decision_function(feats)[0])
    p = score_to_percent_notspoof(s, meta.enroll_scores_sorted or [])
    return s, p
