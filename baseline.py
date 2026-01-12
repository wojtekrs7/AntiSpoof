from __future__ import annotations
from pathlib import Path
import numpy as np
import joblib
from sklearn.mixture import GaussianMixture
import librosa
from scipy.fftpack import dct
from .features import extract_features_from_wav


def lfcc(y: np.ndarray, sr: int = 16000, n_fft: int = 512,
         hop_length: int = 160, win_length: int = 400,
         n_lin: int = 70, n_ceps: int = 20,
         fmin: float = 0.0, fmax: float | None = None,
         eps: float = 1e-10) -> np.ndarray:

    if fmax is None:
        fmax = sr / 2

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                            win_length=win_length, window="hann"))**2

    freqs = np.linspace(0, sr/2, 1 + n_fft//2)

    edges = np.linspace(fmin, fmax, n_lin + 2)

    fb = np.zeros((n_lin, len(freqs)), dtype=np.float32)
    for i in range(n_lin):
        l, c, r = edges[i], edges[i+1], edges[i+2]
        left = np.logical_and(freqs >= l, freqs <= c)
        fb[i, left] = (freqs[left] - l) / max(c - l, 1e-9)
        right = np.logical_and(freqs >= c, freqs <= r)
        fb[i, right] = (r - freqs[right]) / max(r - c, 1e-9)

    F = np.dot(fb, S[:fb.shape[1], :]) + eps

    C = dct(np.log(F), type=2, axis=0, norm="ortho")[:n_ceps, :]
    return C.astype(np.float32)

def lfcc_from_path(path: Path, sr: int = 16000) -> np.ndarray:
    y, s = librosa.load(str(path), sr=sr, mono=True)
    return lfcc(y, sr=s)

def lfcc_stats_feat(path: Path) -> np.ndarray:
    C = lfcc_from_path(path)  # (n_ceps, T)
    mu = np.mean(C, axis=1)
    sd = np.std(C, axis=1) + 1e-8
    return np.concatenate([mu, sd]).astype(np.float32)

def mfcc_stats_feat(path: Path) -> np.ndarray:
    vec = extract_features_from_wav(path, feature_types="mfcc")
    return vec.astype(np.float32)


def cqcc_stats_feat(path: Path) -> np.ndarray:
    vec = extract_features_from_wav(path, feature_types="cqcc")
    return vec.astype(np.float32)



class LFCCGMM:
    def __init__(self, n_components: int = 512, covariance_type: str = "diag",
                 reg_covar: float = 1e-6, random_state: int = 0):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.gmm_genuine = None
        self.gmm_spoof = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xg, Xs = X[y == 0], X[y == 1]
        n_g, n_s = Xg.shape[0], Xs.shape[0]

        if n_g == 0 or n_s == 0:
            raise ValueError(
                f"LFCCGMM.fit: jedna z klas jest pusta: genuine={n_g}, spoof={n_s}."
            )
        n_components = min(self.n_components, n_g, n_s)
        if n_components < self.n_components:
            print(
                f"[WARN] LFCC-GMM: zmniejszam n_components z {self.n_components} do "
                f"{n_components} (genuine={n_g}, spoof={n_s})."
            )

        self.gmm_genuine = GaussianMixture(
            n_components,
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar,
            random_state=self.random_state,
        ).fit(Xg)

        self.gmm_spoof = GaussianMixture(
            n_components,
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar,
            random_state=self.random_state,
        ).fit(Xs)

        self.n_components_ = n_components
        return self

    @property
    def n_features_in_(self):
        return self.gmm_genuine.means_.shape[1]

    def _loglikes(self, X: np.ndarray) -> np.ndarray:
        if self.gmm_genuine is None or self.gmm_spoof is None:
            raise RuntimeError("Modele GMM nie są wytrenowane (gmm_genuine/gmm_spoof = None).")

        logp_g = self.gmm_genuine.score_samples(X)  # (n,)
        logp_s = self.gmm_spoof.score_samples(X)    # (n,)
        return np.stack([logp_g, logp_s], axis=1)   # (n, 2)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logp = self._loglikes(X)                  # (n, 2)
        logp = logp - logp.max(axis=1, keepdims=True)  # stabilizacja numeryczna
        p = np.exp(logp)
        p = p / p.sum(axis=1, keepdims=True)
        return p

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        lg = self.gmm_genuine.score_samples(X)
        ls = self.gmm_spoof.score_samples(X)
        return (lg - ls)  # LLR

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) < 0).astype(int)  # 1=SPOOF

def _scan(data_dir: Path):
    from .dataset import iter_wavs_with_labels
    return list(iter_wavs_with_labels(data_dir))

def extract_lfcc_dataset(data_dir: Path, max_files: int | None = None):
    files = _scan(data_dir)
    if not files:
        raise RuntimeError(f"Brak plików WAV w {data_dir}/genuine i {data_dir}/spoof")

    total = len(files)
    print(f"[INFO] LFCC-GMM: znaleziono {total} nagrań (genuine + spoof).")

    do_balance = False
    target_per_class = None

    idx_order = np.arange(total)

    if max_files is not None and max_files > 0 and max_files < total:
        labels = np.array([lab for (_, lab) in files], dtype=int)
        n_g = int(np.sum(labels == 0))
        n_s = int(np.sum(labels == 1))
        print(f"[INFO] baseline: w zbiorze: genuine={n_g}, spoof={n_s}")

        rng = np.random.default_rng(0)
        rng.shuffle(idx_order)

        if n_g > 0 and n_s > 0:
            target_per_class = min(max_files // 2, n_g, n_s)
            if target_per_class <= 0:
                print(
                    "[WARN] baseline: zbyt mało którejś klasy – "
                    "rezygnuję z balansowania; max_files będzie traktowane globalnie."
                )
            else:
                do_balance = True
                if 2 * target_per_class < max_files:
                    print(
                        f"[WARN] baseline: nie da się wziąć {max_files} zbalansowanych próbek; "
                        f"użyję maks. {2 * target_per_class} "
                        f"({target_per_class} genuine + {target_per_class} spoof)."
                    )
                print(
                    f"[INFO] baseline: balansowanie klas: cel = "
                    f"{target_per_class} genuine + {target_per_class} spoof."
                )
        else:
            print(
                "[WARN] baseline: tylko jedna klasa w danych – "
                "max_files będzie stosowane bez balansowania klas."
            )
    else:
        if max_files is not None and max_files > 0 and max_files >= total:
            print(
                f"[INFO] baseline: podany limit ({max_files}) >= liczby plików, "
                f"używam wszystkich {total}."
            )
        else:
            print("[INFO] baseline: używam wszystkich nagrań (bez balansowania klas).")

    X, y = [], []
    skipped = 0
    used_total = 0
    used_g = 0
    used_s = 0
    attempts = 0

    for pos, i in enumerate(idx_order, start=1):
        p, lab = files[i]
        attempts += 1
        if do_balance:
            if lab == 0 and used_g >= target_per_class:
                continue
            if lab == 1 and used_s >= target_per_class:
                continue
        elif max_files is not None and max_files > 0 and used_total >= max_files:
            break

        try:
            feat = lfcc_stats_feat(p)  # (n_ceps*4,)
            X.append(feat)
            y.append(lab)
            used_total += 1
            if lab == 0:
                used_g += 1
            else:
                used_s += 1
        except Exception as e:
            skipped += 1
            print(
                f"[WARN] baseline: pomijam plik {p.name} – nie udało się policzyć LFCC: {e}",
                flush=True,
            )

        percent = used_total * 100.0 / (2 * target_per_class if do_balance else (max_files or total))
        print(
            f"[PROGRESS] CQCC baseline: used={used_total} "
            f"(g={used_g}, s={used_s})  ({percent:5.1f}%)  ostatni: {p.name}"
        )
        if do_balance and used_g >= target_per_class and used_s >= target_per_class:
            break

    if skipped > 0:
        print(f"[INFO] LFCC-GMM: pominięto {skipped} problematycznych plików.")

    if not X:
        raise RuntimeError("Nie udało się policzyć LFCC dla żadnego pliku.")

    print(
        f"[INFO] LFCC-GMM: zebrano {used_total} przykładów "
        f"(genuine={used_g}, spoof={used_s})."
    )

    return np.vstack(X).astype(np.float32), np.array(y, dtype=np.int64)

def extract_cqcc_dataset(data_dir: Path, max_files: int | None = None):
    files = _scan(data_dir)
    if not files:
        raise RuntimeError(f"Brak plików WAV w {data_dir}/genuine i {data_dir}/spoof")

    total = len(files)
    print(f"[INFO] CQCC-GMM: znaleziono {total} nagrań (genuine + spoof).")
    do_balance = False
    target_per_class = None

    idx_order = np.arange(total)

    if max_files is not None and max_files > 0 and max_files < total:
        labels = np.array([lab for (_, lab) in files], dtype=int)
        n_g = int(np.sum(labels == 0))
        n_s = int(np.sum(labels == 1))
        print(f"[INFO] baseline: w zbiorze: genuine={n_g}, spoof={n_s}")

        rng = np.random.default_rng(0)
        rng.shuffle(idx_order)

        if n_g > 0 and n_s > 0:
            target_per_class = min(max_files // 2, n_g, n_s)
            if target_per_class <= 0:
                print(
                    "[WARN] baseline: zbyt mało którejś klasy – "
                    "rezygnuję z balansowania; max_files będzie traktowane globalnie."
                )
            else:
                do_balance = True
                if 2 * target_per_class < max_files:
                    print(
                        f"[WARN] baseline: nie da się wziąć {max_files} zbalansowanych próbek; "
                        f"użyję maks. {2 * target_per_class} "
                        f"({target_per_class} genuine + {target_per_class} spoof)."
                    )
                print(
                    f"[INFO] baseline: balansowanie klas: cel = "
                    f"{target_per_class} genuine + {target_per_class} spoof."
                )
        else:
            print(
                "[WARN] baseline: tylko jedna klasa w danych – "
                "max_files będzie stosowane bez balansowania klas."
            )
    else:
        if max_files is not None and max_files > 0 and max_files >= total:
            print(
                f"[INFO] baseline: podany limit ({max_files}) >= liczby plików, "
                f"używam wszystkich {total}."
            )
        else:
            print("[INFO] baseline: używam wszystkich nagrań (bez balansowania klas).")

    X, y = [], []
    skipped = 0
    used_total = 0
    used_g = 0
    used_s = 0
    attempts = 0

    for pos, i in enumerate(idx_order, start=1):
        p, lab = files[i]
        attempts += 1

        if do_balance:
            if lab == 0 and used_g >= target_per_class:
                continue
            if lab == 1 and used_s >= target_per_class:
                continue
        elif max_files is not None and max_files > 0 and used_total >= max_files:
            break

        try:
            feat = cqcc_stats_feat(p)
            X.append(feat)
            y.append(lab)
            used_total += 1
            if lab == 0:
                used_g += 1
            else:
                used_s += 1
        except Exception as e:
            skipped += 1
            print(
                f"[WARN] baseline: pomijam plik {p.name} – nie udało się policzyć CQCC: {e}",
                flush=True,
            )

        percent = attempts * 100.0 / total
        if attempts % 1000 == 0 or attempts == total:
            print(
                f"[PROGRESS] CQCC baseline: {attempts}/{total} prób ({percent:5.1f}%)  "
                f"ostatni: {p.name}",
                flush=True,
            )

        if do_balance and used_g >= target_per_class and used_s >= target_per_class:
            break

    if skipped > 0:
        print(f"[INFO] CQCC-GMM: pominięto {skipped} problematycznych plików.")

    if not X:
        raise RuntimeError("Nie udało się policzyć CQCC dla żadnego pliku.")

    print(
        f"[INFO] CQCC-GMM: zebrano {used_total} przykładów "
        f"(genuine={used_g}, spoof={used_s})."
    )

    return np.vstack(X).astype(np.float32), np.array(y, dtype=np.int64)

def extract_mfcc_dataset(data_dir: Path, max_files: int | None = None):
    files = _scan(data_dir)
    if not files:
        raise RuntimeError(f"Brak plików WAV w {data_dir}/genuine i {data_dir}/spoof")

    total = len(files)
    print(f"[INFO] MFCC-GMM: znaleziono {total} nagrań (genuine + spoof).")
    do_balance = False
    target_per_class = None

    idx_order = np.arange(total)

    if max_files is not None and max_files > 0 and max_files < total:
        labels = np.array([lab for (_, lab) in files], dtype=int)
        n_g = int(np.sum(labels == 0))
        n_s = int(np.sum(labels == 1))
        print(f"[INFO] baseline: w zbiorze: genuine={n_g}, spoof={n_s}")

        rng = np.random.default_rng(0)
        rng.shuffle(idx_order)

        if n_g > 0 and n_s > 0:
            target_per_class = min(max_files // 2, n_g, n_s)
            if target_per_class <= 0:
                print(
                    "[WARN] baseline: zbyt mało którejś klasy – "
                    "rezygnuję z balansowania; max_files będzie traktowane globalnie."
                )
            else:
                do_balance = True
                if 2 * target_per_class < max_files:
                    print(
                        f"[WARN] baseline: nie da się wziąć {max_files} zbalansowanych próbek; "
                        f"użyję maks. {2 * target_per_class} "
                        f"({target_per_class} genuine + {target_per_class} spoof)."
                    )
                print(
                    f"[INFO] baseline: balansowanie klas: cel = "
                    f"{target_per_class} genuine + {target_per_class} spoof."
                )
        else:
            print(
                "[WARN] baseline: tylko jedna klasa w danych – "
                "max_files będzie stosowane bez balansowania klas."
            )
    else:
        if max_files is not None and max_files > 0 and max_files >= total:
            print(
                f"[INFO] baseline: podany limit ({max_files}) >= liczby plików, "
                f"używam wszystkich {total}."
            )
        else:
            print("[INFO] baseline: używam wszystkich nagrań (bez balansowania klas).")

    X, y = [], []
    skipped = 0
    used_total = 0
    used_g = 0
    used_s = 0
    attempts = 0

    for pos, i in enumerate(idx_order, start=1):
        p, lab = files[i]
        attempts += 1

        if do_balance:
            if lab == 0 and used_g >= target_per_class:
                continue
            if lab == 1 and used_s >= target_per_class:
                continue
        elif max_files is not None and max_files > 0 and used_total >= max_files:
            break

        try:
            feat = mfcc_stats_feat(p)
            X.append(feat)
            y.append(lab)
            used_total += 1
            if lab == 0:
                used_g += 1
            else:
                used_s += 1
        except Exception as e:
            skipped += 1
            print(
                f"[WARN] baseline: pomijam plik {p.name} – nie udało się policzyć MFCC: {e}",
                flush=True,
            )

        percent = attempts * 100.0 / total
        if attempts % 1000 == 0 or attempts == total:
            print(
                f"[PROGRESS] MFCC baseline: {attempts}/{total} prób ({percent:5.1f}%)  "
                f"ostatni: {p.name}",
                flush=True,
            )

        if do_balance and used_g >= target_per_class and used_s >= target_per_class:
            break

    if skipped > 0:
        print(f"[INFO] MFCC-GMM: pominięto {skipped} problematycznych plików.")

    if not X:
        raise RuntimeError("Nie udało się policzyć MFCC dla żadnego pliku.")

    print(
        f"[INFO] MFCC-GMM: zebrano {used_total} przykładów "
        f"(genuine={used_g}, spoof={used_s})."
    )

    return np.vstack(X).astype(np.float32), np.array(y, dtype=np.int64)


def train_lfcc_gmm(data_dir: Path, model_path: Path, max_files: int | None = None):
    X, y = extract_lfcc_dataset(data_dir, max_files=max_files)
    clf = LFCCGMM().fit(X, y)
    joblib.dump(clf, model_path)
    return X.shape, model_path

def predict_lfcc_gmm(model_path: Path, wav_path: Path):
    clf: LFCCGMM = joblib.load(model_path)
    x = lfcc_stats_feat(Path(wav_path)).reshape(1, -1)
    yhat = int(clf.predict(x)[0])
    llr = float(clf.decision_function(x)[0])
    label = "SPOOF" if yhat == 1 else "GENUINE"
    return label, llr

def train_cqcc_gmm(data_dir: Path, model_path: Path, max_files: int | None = None):
    X, y = extract_cqcc_dataset(data_dir, max_files=max_files)
    clf = LFCCGMM().fit(X, y)  # ta sama klasa, inny wektor cech
    joblib.dump(clf, model_path)
    return X.shape, model_path


def predict_cqcc_gmm(model_path: Path, wav_path: Path):
    clf: LFCCGMM = joblib.load(model_path)
    x = cqcc_stats_feat(Path(wav_path)).reshape(1, -1)
    yhat = int(clf.predict(x)[0])
    llr = float(clf.decision_function(x)[0])
    label = "SPOOF" if yhat == 1 else "GENUINE"
    return label, llr

def train_mfcc_gmm(data_dir: Path, model_path: Path, max_files: int | None = None):
    X, y = extract_mfcc_dataset(data_dir, max_files=max_files)
    clf = LFCCGMM().fit(X, y)
    joblib.dump(clf, model_path)
    return X.shape, model_path


def predict_mfcc_gmm(model_path: Path, wav_path: Path):
    clf: LFCCGMM = joblib.load(model_path)
    x = mfcc_stats_feat(Path(wav_path)).reshape(1, -1)
    yhat = int(clf.predict(x)[0])
    llr = float(clf.decision_function(x)[0])
    label = "SPOOF" if yhat == 1 else "GENUINE"
    return label, llr
