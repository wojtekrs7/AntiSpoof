from __future__ import annotations

import argparse
from pathlib import Path
import sys
from datetime import datetime
import numpy as np
import joblib
import warnings
from .features import extract_features_from_wav
from .baseline import LFCCGMM, lfcc_stats_feat
from .baseline import (
    train_lfcc_gmm,
    predict_lfcc_gmm,
    train_mfcc_gmm,
    predict_mfcc_gmm,
    train_cqcc_gmm,
    predict_cqcc_gmm,
)


from .features import extract_features_from_wav
from .dataset import iter_wavs_with_labels
from .detector import train_model
from .oneclass import (
    fit_enroll,
    save_enroll,
    load_enroll,
    predict_percent_notspoof,
)

ARTIFACTS_DIR = Path("artifacts")
FEAT_DIR = ARTIFACTS_DIR / "features"
MODEL_DIR = ARTIFACTS_DIR / "models"


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dirs():
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _latest_npz(dir_path: Path) -> Path | None:
    files = sorted(dir_path.glob("*.npz"))
    return files[-1] if files else None


# =============================================================================
#  TRAIN (BINARY: SVM / LOGREG / RF)
# =============================================================================

def _cmd_train(args):
    _ensure_dirs()
    print(f"[INFO] Trening modelu {args.model}, cechy = {args.feat}")

    # --- zbierz wszystkie pliki, żeby znać ich liczbę ---
    items = list(iter_wavs_with_labels(args.data_dir))
    total = len(items)
    if total == 0:
        print(f"[ERROR] Nie znaleziono żadnych plików .wav w {args.data_dir}")
        return

    print(f"[INFO] Znaleziono {total} nagrań (genuine + spoof).")

    # limit liczby plików, jeśli podany (losowy wybór, zbalansowany 50/50) ---
    if getattr(args, "max_files", None) is not None and args.max_files > 0:
        if args.max_files < total:
            # podzial na klasy
            genuine = [it for it in items if it[1] == 0]
            spoof   = [it for it in items if it[1] == 1]
            n_g = len(genuine)
            n_s = len(spoof)

            print(f"[INFO] W zbiorze: genuine={n_g}, spoof={n_s}")

            if n_g == 0 or n_s == 0:
                print("[WARN] Tylko jedna klasa w danych – losuję z całej puli.")
                idx = np.arange(total)
                rng = np.random.default_rng(0)
                rng.shuffle(idx)
                selected_idx = idx[:args.max_files]
                items = [items[i] for i in selected_idx]
                total = len(items)
            else:
                n_per_class = args.max_files // 2
                n = min(n_per_class, n_g, n_s)

                if 2 * n < args.max_files:
                    print(
                        f"[WARN] Nie da się wziąć {args.max_files} plików "
                        f"(za mało którejś klasy). Użyję {2*n} nagrań "
                        f"({n} genuine + {n} spoof)."
                    )

                rng = np.random.default_rng(0)
                rng.shuffle(genuine)
                rng.shuffle(spoof)

                selected = genuine[:n] + spoof[:n]
                rng.shuffle(selected)  # pomieszaj klasy

                items = selected
                total = len(items)
                print(
                    f"[INFO] Zbalansowany podzbiór do treningu: "
                    f"{n} genuine + {n} spoof = {total} nagrań."
                )
        else:
            print(
                f"[INFO] Podany limit ({args.max_files}) >= liczby plików, "
                f"używam wszystkich {total}."
            )
    else:
        print("[INFO] Używam wszystkich nagrań.")

    print("[INFO] Start ekstrakcji cech.")

    X_list = []
    y_list = []
    skipped = 0

    for idx, (wav_path, label) in enumerate(items, start=1):
        try:
            feats = extract_features_from_wav(
                wav_path,
                feature_types=args.feat,
            )
        except Exception as e:
            skipped += 1
            print(
                f"[WARN] Pomijam plik {wav_path.name} – nie udało się wczytać: {e}",
                flush=True,
            )
            continue

        X_list.append(feats)
        y_list.append(label)

        if idx % 100 == 0 or idx == total:
            percent = idx * 100.0 / total
            print(
                f"[PROGRESS] Ekstrakcja cech: {idx}/{total} "
                f"({percent:5.1f}%)  ostatni: {wav_path.name}",
                flush=True,
            )

    if not X_list:
        print("[ERROR] Wszystkie pliki zostały odrzucone (brak przykładów do treningu).")
        return

    if skipped > 0:
        print(f"[INFO] Pominięto {skipped} uszkodzonych / nieobsługiwanych plików.")

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    unique_classes = np.unique(y)
    if unique_classes.size < 2:
        print(f"[ERROR] W wybranej próbce jest tylko jedna klasa: {unique_classes}.")
        print("[HINT] Zwieksz --max-files albo sprawdź strukturę katalogow genuine/spoof.")
        return

    print(f"[INFO] Zestaw treningowy: X={X.shape}, y={y.shape}")
    print(f"[INFO] Uczenie modelu {args.model}.")

    clf = train_model(X, y, model_name=args.model)

    out_path = MODEL_DIR / f"{args.model}_{args.feat}_{_ts()}.joblib"
    joblib.dump(clf, out_path)
    print(f"[OK] Zapisano model: {out_path}")

def _cmd_predict(args):
    model_path = Path(args.model)
    wav_path = Path(args.wav)

    if not model_path.is_file():
        print(f"[ERROR] Nie znaleziono modelu: {model_path}")
        return

    if not wav_path.is_file():
        print(f"[ERROR] Nie znaleziono pliku audio: {wav_path}")
        return

    print(f"[INFO] Ładuję model: {model_path}")
    clf = joblib.load(model_path)

    print(f"[INFO] Ekstrakcja cech z: {wav_path}")

    if isinstance(clf, LFCCGMM):
        feats = lfcc_stats_feat(wav_path)
        print("[DEBUG] używam lfcc_stats_feat (LFCC-GMM)")
    else:
        feats = extract_features_from_wav(
            wav_path,
            feature_types=args.feat,
        )

    feats = np.asarray(feats, dtype=np.float32)

    if feats.ndim == 1:
        feats = feats.reshape(1, -1)
    elif feats.ndim > 2:
        feats = feats.reshape(feats.shape[0], -1)

    print(f"[DEBUG] feats.shape po reshape = {feats.shape}")
    from sklearn.pipeline import Pipeline

    n_expected = None
    if isinstance(clf, Pipeline) and "scaler" in clf.named_steps:
        n_expected = clf.named_steps["scaler"].n_features_in_
    elif hasattr(clf, "n_features_in_"):
        n_expected = clf.n_features_in_

    print(f"[DEBUG] model oczekuje: {n_expected} cech")

    if n_expected is not None and feats.shape[1] != n_expected:
        print(f"[WARN] Model oczekuje {n_expected} cech, ale policzono {feats.shape[1]}.")

        if feats.shape[1] > n_expected:
            feats = feats[:, :n_expected]
            print(f"[INFO] Przyciąłem cechy do: {feats.shape}")
        else:
            padded = np.zeros((feats.shape[0], n_expected), dtype=feats.dtype)
            padded[:, :feats.shape[1]] = feats
            feats = padded
            print(f"[INFO] Dopełniłem cechy zerami do: {feats.shape}")
    proba_spoof = clf.predict_proba(feats)[0, 1]
    proba_bona = 1.0 - proba_spoof

    label = "spoof" if proba_spoof >= 0.5 else "bonafide"

    print("")
    print(f"Wynik dla pliku: {wav_path}")
    print(f"  P(bonafide) = {proba_bona:.3f}")
    print(f"  P(spoof)    = {proba_spoof:.3f}")
    print(f"  Decyzja:    {label}")

# =============================================================================
#  ONE-CLASS: ENROLL
# =============================================================================
def _cmd_oc_train(args):
    _ensure_dirs()

    print(
        f"[INFO] Trening one-class {args.oc_model}, "
        f"cechy = {args.feat}, data_dir = {args.data_dir}"
    )

    all_items = list(iter_wavs_with_labels(args.data_dir))
    genuine = [(p, y) for (p, y) in all_items if y == 0]
    total = len(genuine)

    if total == 0:
        print(
            f"[ERROR] Nie znaleziono żadnych nagrań klasy genuine (etykieta 0) "
            f"w {args.data_dir}/genuine."
        )
        sys.exit(1)

    print(f"[INFO] Znaleziono {total} nagrań genuine do treningu one-class.")

    items = genuine
    if args.max_files is not None and args.max_files > 0:
        if args.max_files < total:
            n = args.max_files
            print(
                f"[INFO] Używam losowego podzbioru {n} z {total} nagrań genuine "
                f"do treningu one-class."
            )
            idx = np.arange(total)
            rng = np.random.default_rng(0)
            rng.shuffle(idx)
            selected_idx = idx[:n]
            items = [items[i] for i in selected_idx]
            total = len(items)
        else:
            print(
                f"[INFO] Podany limit ({args.max_files}) >= liczby plików, "
                f"używam wszystkich {total} genuine."
            )
    else:
        print("[INFO] Używam wszystkich nagrań genuine.")

    print("[INFO] Start ekstrakcji cech (one-class).")

    X_list = []
    skipped = 0

    for idx, (wav_path, _) in enumerate(items, start=1):
        try:
            feats = extract_features_from_wav(
                wav_path,
                feature_types=args.feat,
            )
        except Exception as e:
            skipped += 1
            print(
                f"[WARN] Pomijam plik {wav_path.name} – nie udało się wczytać: {e}",
                flush=True,
            )
            continue

        X_list.append(feats)

        if idx % 100 == 0 or idx == total:
            percent = idx * 100.0 / total
            print(
                f"[PROGRESS] Ekstrakcja cech (OC): {idx}/{total} "
                f"({percent:5.1f}%)  ostatni: {wav_path.name}",
                flush=True,
            )

    if not X_list:
        print("[ERROR] Nie udało się policzyć cech dla żadnego pliku genuine.")
        sys.exit(1)

    feats = np.asarray(X_list, dtype=np.float32)

    if feats.ndim == 1:
        feats = feats.reshape(1, -1)
    elif feats.ndim > 2:
        feats = feats.reshape(feats.shape[0], -1)

    print(f"[DEBUG] OC feats.shape po reshape = {feats.shape}")

    model, meta = fit_enroll(feats, model_name=args.oc_model)
    meta.feat = args.feat

    ts = _ts()
    out_model = MODEL_DIR / f"oc_{args.oc_model}_{args.feat}_{_ts()}.joblib"
    save_enroll(model, meta, out_model)
    print(f"[OK] Zapisano model one-class (OC): {out_model}")




def _cmd_oc_enroll(args):
    _ensure_dirs()

    exts = ("*.wav", "*.flac")
    enroll_paths = []
    for ext in exts:
        enroll_paths.extend(Path(args.enroll_dir).rglob(ext))
    enroll_paths = sorted(set(enroll_paths))  # na wszelki wypadek

    if not enroll_paths:
        print("[ERR] Brak plików WAV/FLAC do enrollingu.")
        sys.exit(1)

    print(f"[INFO] One-Class ENROLL, model={args.oc_model}, cechy = {args.feat}")

    feats_all = []
    for p in enroll_paths:
        vec = extract_features_from_wav(
            p,
            feature_types=args.feat,
        )
        feats_all.append(vec)

    feats_all = np.vstack(feats_all).astype(np.float32)

    model, meta = fit_enroll(feats_all, model_name=args.oc_model)
    meta.feat = args.feat

    out_model = MODEL_DIR / f"oc_{args.oc_model}_{args.feat}_{_ts()}.joblib"

    save_enroll(model, meta, out_model)

    print(f"[OK] Zapisano model OC (model+meta): {out_model}")

def _cmd_oc_predict(args):
    model, meta = load_enroll(args.model)

    feat_type = getattr(meta, "feat", "mfcc")
    print(
        f"[INFO] One-Class PREDICT, "
        f"model={getattr(meta, 'oc_model_name', 'unknown')}, "
        f"feat={feat_type}"
    )

    score, percent = predict_percent_notspoof(model, meta, args.wav)

    print(f"[RESULT] score={score:.4f}, percent_notspoof={percent:.2f}%")

def _cmd_baseline_train(args):
    _ensure_dirs()

    feat_type = getattr(args, "feat_type", "lfcc")
    print(f"[INFO] Trening baseline {feat_type.upper()}-GMM na danych z: {args.data_dir}")
    if getattr(args, "max_files", None):
        print(f"[INFO] baseline: max_files = {args.max_files}")

    ts = _ts()
    if feat_type == "mfcc":
        out_path = MODEL_DIR / f"mfcc_gmm_{ts}.joblib"
        shape, model_path = train_mfcc_gmm(
            args.data_dir,
            out_path,
            max_files=getattr(args, "max_files", None),
        )
    elif feat_type == "cqcc":
        out_path = MODEL_DIR / f"cqcc_gmm_{ts}.joblib"
        shape, model_path = train_cqcc_gmm(
            args.data_dir,
            out_path,
            max_files=getattr(args, "max_files", None),
        )
    else:
        out_path = MODEL_DIR / f"lfcc_gmm_{ts}.joblib"
        shape, model_path = train_lfcc_gmm(
            args.data_dir,
            out_path,
            max_files=getattr(args, "max_files", None),
        )

    print(f"[OK] Zapisano baseline {feat_type.upper()}-GMM: {model_path} (X shape={shape})")


def _cmd_baseline_predict(args):
    feat_type = getattr(args, "feat_type", "lfcc")

    if feat_type == "mfcc":
        label_pred, llr = predict_mfcc_gmm(args.model, args.wav)
    elif feat_type == "cqcc":
        label_pred, llr = predict_cqcc_gmm(args.model, args.wav)
    else:
        label_pred, llr = predict_lfcc_gmm(args.model, args.wav)

    print(f"[RESULT] label={label_pred}, llr={llr:.4f}")

def main(argv=None):
    """
    Komendy:
    - svm-train            – trening klasyfikatora binarnego (SVM),
    - svm-predict          – predykcja modelu binarnego,
    - oc-train              - trenowanie modelu jednoklasowego
    - oc-predict       – predykcja modelu one-class (percent_notspoof),
    - gmm-train   – trening GMM,
    - gmm-predict – predykcja GMM.
    """
    p = argparse.ArgumentParser(prog="antispoof", description="AntiSpoof CLI")
    subparsers = p.add_subparsers(dest="cmd", required=True)

    # -------------------------------------------------------------------------
    # TRAIN (binary)
    p_train = subparsers.add_parser("svm-train", help="Trening binarnego klasyfikatora")
    p_train.add_argument("data_dir", type=Path)
    p_train.add_argument(
        "--feat",
        default="mfcc",
        choices=[
            "mfcc",
            "lfcc",
            "cqcc"
        ],
        help="Zestaw cech (domyślnie: mfcc)",
    )
    p_train.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maksymalna liczba nagrań do treningu (domyślnie: wszystkie)",
    )
    p_train.add_argument(
        "--model",
        default="svm",
        choices=["svm"],
        help="Klasyfikator: svm (domyślnie)",
    )
    p_train.set_defaults(func=_cmd_train)

    # -------------------------------------------------------------------------
    # PREDICT (binary)
    p_pred = subparsers.add_parser("predict", help="Predykcja binarnego klasyfikatora")
    p_pred.add_argument("model", type=Path)
    p_pred.add_argument("wav", type=Path)
    p_pred.add_argument(
        "--feat",
        default="mfcc",
        choices=[
            "mfcc",
            "lfcc",
            "cqcc",
        ],
        help="Zestaw cech (musi być zgodny z tym z treningu)",
    )
    p_pred.set_defaults(func=_cmd_predict)

    p_oc_train = subparsers.add_parser(
        "oc-train",
        help="Trening one-class CM (OC-SVM) na klasie genuine z data_dir",
    )
    p_oc_train.add_argument("data_dir", type=Path)
    p_oc_train.add_argument(
        "--feat",
        default="mfcc",
        choices=[
            "mfcc",
            "lfcc",
            "cqcc",
        ],
        help="Zestaw cech używany przy treningu one-class",
    )
    p_oc_train.add_argument(
        "--oc-model",
        default="ocsvm",
        choices=["ocsvm"],
        help="Model one-class: ocsvm (domyślnie)",
    )
    p_oc_train.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maksymalna liczba nagrań genuine do treningu (domyślnie: wszystkie)",
    )
    p_oc_train.set_defaults(func=_cmd_oc_train)

    p_oc_pred = subparsers.add_parser("oc-predict", help="Predykcja One-Class")
    p_oc_pred.add_argument("model", type=Path)
    p_oc_pred.add_argument("wav", type=Path)
    p_oc_pred.set_defaults(func=_cmd_oc_predict)


    p_btrain = subparsers.add_parser("gmm-train", help="Trening baseline GMM")
    p_btrain.add_argument("data_dir", type=Path)
    p_btrain.add_argument(
        "--feat-type",
        default="lfcc",
        choices=["lfcc", "mfcc", "cqcc"],
        help="Typ cech dla baseline GMM (lfcc/mfcc/cqcc; domyślnie lfcc)",
    )
    p_btrain.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maksymalna liczba nagrań do treningu (domyślnie: wszystkie)",
    )
    p_btrain.set_defaults(func=_cmd_baseline_train)

    p_bpred = subparsers.add_parser("gmm-predict", help="Predykcja baseline GMM")
    p_bpred.add_argument("model", type=Path)
    p_bpred.add_argument("wav", type=Path)
    p_bpred.add_argument(
        "--feat-type",
        default="lfcc",
        choices=["lfcc", "mfcc", "cqcc"],
        help="Typ cech, którego użyto przy trenowaniu modelu (lfcc/mfcc/cqcc).",
    )
    p_bpred.set_defaults(func=_cmd_baseline_predict)

    # -------------------------------------------------------------------------

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()