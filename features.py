from __future__ import annotations
from pathlib import Path
from typing import Iterable, List
import numpy as np
import soundfile as sf
import librosa
from scipy.fftpack import dct
import warnings

def _load_wav_mono(path: Path, sr_target: int = 16000):
    """
    Wczytuje plik audio jako mono, float32, przeskalowany do sr_target.
    Jesli nie da się wczytac – rzuca RuntimeError
    """
    # 1. próba: soundfile
    try:
        x, sr = sf.read(str(path), always_2d=False)
    except Exception as e_sf:
        warnings.warn(
            f"PySoundFile failed for {path.name} ({e_sf}). "
            f"Trying librosa/audioread instead."
        )
    else:
        # mono
        if x.ndim > 1:
            x = np.mean(x, axis=1)
        # resampling
        if sr != sr_target:
            x = librosa.resample(x, orig_sr=sr, target_sr=sr_target)
            sr = sr_target
        return x.astype(np.float32), sr

    # 2. próba: librosa (audioread / ffmpeg backend)
    try:
        x, sr = librosa.load(str(path), sr=sr_target, mono=True)
    except Exception as e_lb:
        warnings.warn(
            f"Nie udalo się wczytac pliku {path} ani przez soundfile, "
            f"ani przez librosa/audioread: {e_lb}"
        )
        raise RuntimeError(f"Could not load audio file: {path}") from e_lb

    return x.astype(np.float32), sr


def _stack_with_deltas(coeffs: np.ndarray,
                       include_deltas: bool = True,
                       include_delta_delta: bool = True) -> np.ndarray:
    """
    coeffs: (n_coeffs, T)
    Zwraca [static; delta; delta-delta] w pionie.
    """
    feats = [coeffs]
    if include_deltas:
        d1 = librosa.feature.delta(coeffs)
        feats.append(d1)
    if include_delta_delta:
        d2 = librosa.feature.delta(coeffs, order=2)
        feats.append(d2)
    return np.vstack(feats)


def _stats_pooling(feats: np.ndarray) -> np.ndarray:
    """
    feats: (D, T)
    Zwraca wektor [mean, std, skew, kurt] o wymiarze 4*D.

    Statistics pooling – zamiast sekwencji w czasie
    otrzymujemy pojedynczy wektor reprezentujacy caly klip.
    """
    mean = np.mean(feats, axis=1)
    std = np.std(feats, axis=1) + 1e-8
    norm = (feats.T - mean) / std
    skew = np.mean(norm ** 3, axis=0)
    kurt = np.mean(norm ** 4, axis=0) - 3.0
    out = np.hstack([mean, std, skew, kurt])
    return out


def _mfcc_feats(x: np.ndarray, sr: int,
                n_mfcc: int = 40,
                include_deltas: bool = True,
                include_delta_delta: bool = True) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc)
    feats = _stack_with_deltas(
        mfcc,
        include_deltas=include_deltas,
        include_delta_delta=include_delta_delta,
    )
    return _stats_pooling(feats)


def _make_lin_filterbank(sr: int,
                         n_fft: int,
                         n_lin: int,
                         fmin: float,
                         fmax: float) -> np.ndarray:

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    edges = np.linspace(fmin, fmax, n_lin + 2)
    fb = np.zeros((n_lin, len(freqs)), dtype=np.float32)

    for i in range(n_lin):
        f_l, f_c, f_r = edges[i], edges[i + 1], edges[i + 2]
        left = np.logical_and(freqs >= f_l, freqs <= f_c)
        if np.any(left):
            fb[i, left] = (freqs[left] - f_l) / max(f_c - f_l, 1e-6)
        right = np.logical_and(freqs >= f_c, freqs <= f_r)
        if np.any(right):
            fb[i, right] = (f_r - freqs[right]) / max(f_r - f_c, 1e-6)

    fb[fb < 0] = 0.0
    return fb


def _lfcc_matrix(x: np.ndarray,
                 sr: int = 16000,
                 n_fft: int = 512,
                 hop_length: int = 160,
                 win_length: int = 400,
                 n_lin: int = 70,
                 n_ceps: int = 20,
                 fmin: float = 0.0,
                 fmax: float | None = None,
                 eps: float = 1e-10) -> np.ndarray:
    if fmax is None:
        fmax = sr / 2.0

    S = librosa.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
    )
    S_power = np.abs(S) ** 2

    fb = _make_lin_filterbank(
        sr=sr,
        n_fft=n_fft,
        n_lin=n_lin,
        fmin=fmin,
        fmax=fmax,
    )
    fb = fb[:, : S_power.shape[0]]

    S_lin = np.dot(fb, S_power)
    S_lin = np.maximum(S_lin, eps)
    log_S = np.log(S_lin)

    ceps = dct(log_S, type=2, axis=0, norm="ortho")[:n_ceps]
    return ceps


def _lfcc_feats(x: np.ndarray, sr: int,
                n_ceps: int = 20,
                include_deltas: bool = True,
                include_delta_delta: bool = True) -> np.ndarray:
    """
    LFCC + delta + delta-delta + statystyki.
    """
    lfcc_mat = _lfcc_matrix(x, sr=sr, n_ceps=n_ceps)
    feats = _stack_with_deltas(
        lfcc_mat,
        include_deltas=include_deltas,
        include_delta_delta=include_delta_delta,
    )
    return _stats_pooling(feats)

def _cqcc_matrix(x: np.ndarray,
                 sr: int = 16000,
                 hop_length: int = 160,
                 fmin: float = 20.0,
                 n_bins: int = 96,
                 bins_per_octave: int = 24,
                 n_ceps: int = 20,
                 eps: float = 1e-10) -> np.ndarray:

    C = librosa.cqt(
        y=x,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
    )
    S_power = np.abs(C) ** 2
    S_power = np.maximum(S_power, eps)
    log_S = np.log(S_power)

    ceps = dct(log_S, type=2, axis=0, norm="ortho")[:n_ceps]
    return ceps


def _cqcc_feats(x: np.ndarray, sr: int,
                n_ceps: int = 20,
                include_deltas: bool = True,
                include_delta_delta: bool = True) -> np.ndarray:
    min_len = 1024
    if len(x) < min_len:
        raise RuntimeError(f"Audio too short for CQCC ({len(x)} samples).")

    cqcc_mat = _cqcc_matrix(x, sr=sr, n_ceps=n_ceps)
    feats = _stack_with_deltas(
        cqcc_mat,
        include_deltas=include_deltas,
        include_delta_delta=include_delta_delta,
    )
    return _stats_pooling(feats)


def extract_features_from_wav(
    path: Path,
    sr_target: int = 16000,
    feature_types: str | Iterable[str] = "mfcc",
    include_deltas: bool = True,
    include_delta_delta: bool = True,
) -> np.ndarray:
    x, sr = _load_wav_mono(path, sr_target=sr_target)

    if isinstance(feature_types, str):
        parts: List[str] = [p.strip().lower() for p in feature_types.split("+")]
    else:
        parts = [str(p).strip().lower() for p in feature_types]

    if not parts:
        raise ValueError("feature_types nie może być puste.")

    all_vecs: List[np.ndarray] = []

    for ft in parts:
        if ft == "mfcc":
            vec = _mfcc_feats(
                x,
                sr,
                include_deltas=include_deltas,
                include_delta_delta=include_delta_delta,
            )
        elif ft == "lfcc":
            vec = _lfcc_feats(
                x,
                sr,
                include_deltas=include_deltas,
                include_delta_delta=include_delta_delta,
            )
        elif ft == "cqcc":
            vec = _cqcc_feats(
                x,
                sr,
                include_deltas=include_deltas,
                include_delta_delta=include_delta_delta,
            )
        else:
            raise ValueError(f"Nieznany feature_type: {ft!r}")

        all_vecs.append(vec.astype(np.float32))

    out = np.concatenate(all_vecs, axis=0).astype(np.float32)
    return out
