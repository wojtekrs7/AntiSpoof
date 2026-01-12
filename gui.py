from __future__ import annotations

import json
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import csv
import joblib
import numpy as np
import sounddevice as sd
import soundfile as sf
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from .features import extract_features_from_wav
from .oneclass import load_enroll, predict_percent_notspoof
from .baseline import lfcc_stats_feat, mfcc_stats_feat, cqcc_stats_feat

import matplotlib
matplotlib.use("Agg")  # zapis wykresów do pliku, bez otwierania okien
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

AUDIO_EXTS = {".wav", ".flac"}
FEAT_CHOICES = [
    "mfcc",
    "lfcc",
    "cqcc",
]


class AntiSpoofGUI(tk.Tk):

    def __init__(self) -> None:
        super().__init__()
        self.title("AntiSpoof — GUI")
        self.geometry("980x640")
        self.minsize(840, 540)

        self.mode = tk.StringVar(value="binary")

        self.feat_var = tk.StringVar(value="mfcc")

        self.bin_model = None
        self.bin_model_path = tk.StringVar(value="(brak)")
        self._bin_model_full: Optional[str] = None

        self.oc_model_path = tk.StringVar(value="(brak)")
        self.oc_enroll_dir = tk.StringVar(value="(nie wybrano)")
        self._oc_model_full: Optional[str] = None
        self._oc_enroll_dir_full: Optional[str] = None
        self.oc_perc_threshold = 90.0

        self.gmm_model = None
        self.gmm_model_path = tk.StringVar(value="(brak)")
        self._gmm_model_full: Optional[str] = None
        self.gmm_feat_var = tk.StringVar(value="lfcc")

        self.audio_path = tk.StringVar(value="(nie wybrano)")

        self.result_var = tk.StringVar(value="—")
        self._last_batch_dir: Optional[str] = None

        self.eval_dir = tk.StringVar(value="(nie wybrano)")
        self.eval_labels = tk.StringVar(value="(nie wybrano)")
        self._eval_dir_full: Optional[str] = None
        self._eval_labels_full: Optional[str] = None

        self._build_ui()
        self._log(
            "Witaj! Wybierz tryb pracy, załaduj odpowiedni model.\n"
            "Plik audio w sekcji „Audio” służy tylko do pojedynczej weryfikacji. "
            "Dla trybu binarnego wybierz zestaw cech zgodny z użytym w treningu."
        )

    def on_choose_eval_dir(self) -> None:
        path = filedialog.askdirectory(title="Wybierz folder z plikami do ewaluacji")
        if not path:
            return
        p = Path(path)
        self._eval_dir_full = str(p.resolve())
        self.eval_dir.set(str(p.resolve()))
        self._log(f"Wybrano folder ewaluacji: {p}")
        self._update_buttons()

    def on_choose_eval_labels(self) -> None:
        path = filedialog.askopenfilename(
            title="Wybierz plik klucza (.txt)",
            filetypes=[("Pliki tekstowe", "*.txt"), ("Wszystkie pliki", "*.*")],
        )
        if not path:
            return
        p = Path(path)
        self._eval_labels_full = str(p.resolve())
        self.eval_labels.set(str(p.resolve()))
        self._log(f"Wybrano plik klucza: {p}")
        self._update_buttons()

    def _parse_label_token(self, tok: str) -> int:
        t = tok.strip().lower()
        if t in ("genuine", "bonafide", "0", "nonspoof"):
            return 0
        if t in ("spoof", "1"):
            return 1
        raise ValueError(f"Nieznana etykieta w pliku klucza: {tok!r}")

    def on_eval_eer(self) -> None:
        if not self._eval_dir_full:
            messagebox.showinfo(
                "Testowanie",
                "Najpierw wybierz folder z plikami testowymi.",
            )
            return

        if not self._eval_labels_full:
            self._log(
                "[INFO] Używam nazw katalogów "
                "(genuine/spoof) do ustalenia etykiet."
            )


        m = self.mode.get()
        if m == "binary" and not self.bin_model:
            messagebox.showinfo("Ewaluacja EER", "Najpierw wczytaj model SVM.")
            return
        if m == "oneclass" and not self._oc_model_full:
            messagebox.showinfo(
                "Ewaluacja EER",
                "Najpierw wczytaj model one-class.",
            )
            return
        if m == "gmm" and not self.gmm_model:
            messagebox.showinfo("Ewaluacja EER", "Najpierw wczytaj model GMM.")
            return

        self.btn_eval.configure(state="disabled")
        self._log("=== Start ewaluacji EER ===")
        threading.Thread(target=self._eval_eer_worker, daemon=True).start()

    def _infer_label_from_path(self, wav_path: Path, root: Path) -> Optional[int]:
        try:
            rel_parts = [p.lower() for p in wav_path.relative_to(root).parts]
        except Exception:
            rel_parts = [p.lower() for p in wav_path.parts]

        if any(x in ("genuine", "bonafide", "nonspoof") for x in rel_parts):
            return 0
        if "spoof" in rel_parts:
            return 1
        return None


    def _eval_eer_worker(self) -> None:
        try:
            eval_dir = Path(self._eval_dir_full)
            mode = self.mode.get()
            labels_path_str = self._eval_labels_full  # może być None

            entries: list[tuple[Path, int]] = []

            if labels_path_str:
                labels_path = Path(labels_path_str)
                with labels_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = line.split()
                        if len(parts) < 2:
                            continue
                        fname, lab_tok = parts[0], parts[1]
                        try:
                            lab = self._parse_label_token(lab_tok)
                        except ValueError as e:
                            self._log(f"[WARN] {e}")
                            continue
                        wav_path = eval_dir / fname
                        if not wav_path.exists():
                            self._log(f"[WARN] Plik z klucza nie istnieje: {wav_path}")
                            continue
                        entries.append((wav_path, lab))
            else:
                self._log(
                    "[INFO] Generuję etykiety na podstawie struktury katalogów "
                    "(szukam 'genuine' vs 'spoof' w ścieżce pliku)."
                )
                for pattern in ("*.wav", "*.flac"):
                    for wav_path in eval_dir.rglob(pattern):
                        if not wav_path.is_file():
                            continue
                        lab = self._infer_label_from_path(wav_path, eval_dir)
                        if lab is None:
                            self._log(
                                f"[WARN] Nie udało się ustalić etykiety dla {wav_path}; "
                                "pomijam (brak 'genuine'/'spoof' w ścieżce)."
                            )
                            continue
                        entries.append((wav_path, lab))

            if not entries:
                self._log("[ERROR] Brak poprawnych przykładów do ewaluacji.")
                return

            self._log(f"[INFO] Ewaluacja na {len(entries)} plikach.")


            if not entries:
                self._log("[ERROR] Brak poprawnych wpisów w pliku klucza.")
                return

            self._log(f"[INFO] Ewaluacja na {len(entries)} plikach.")

            y_true: list[int] = []
            scores: list[float] = []

            if mode == "oneclass":
                model, meta = load_enroll(self._oc_model_full)

            for idx, (wav_path, lab) in enumerate(entries, start=1):
                try:
                    if mode == "binary":
                        feats = extract_features_from_wav(
                            wav_path,
                            feature_types=self.feat_var.get(),
                        ).reshape(1, -1)
                        clf = self.bin_model
                        if hasattr(clf, "decision_function"):
                            s = float(clf.decision_function(feats)[0])
                        elif hasattr(clf, "predict_proba"):
                            proba = clf.predict_proba(feats)[0]
                            s = float(proba[1])
                        else:
                            pred = int(clf.predict(feats)[0])
                            s = float(pred)

                    elif mode == "oneclass":
                        score, perc = predict_percent_notspoof(model, meta, wav_path)
                        s = -float(score)

                    else:  # gmm
                        clf = self.gmm_model
                        feat_type = self.gmm_feat_var.get()
                        if feat_type == "mfcc":
                            x = mfcc_stats_feat(wav_path).reshape(1, -1)
                        elif feat_type == "cqcc":
                            x = cqcc_stats_feat(wav_path).reshape(1, -1)
                        else:  # lfcc
                            x = lfcc_stats_feat(wav_path).reshape(1, -1)

                        llr = float(clf.decision_function(x)[0])
                        s = -llr

                    y_true.append(lab)
                    scores.append(s)

                    if idx % 20 == 0 or idx == len(entries):
                        self._log(
                            f"[PROGRESS] {idx}/{len(entries)}  "
                            f"ostatni: {wav_path.name}  score={s:.4f}"
                        )
                except Exception as e:
                    self._log(f"[WARN] Błąd dla pliku {wav_path.name}: {e}")

            if not y_true:
                self._log("[ERROR] Nie udało się policzyć score'ów dla żadnego pliku.")
                return

            y_true_np = np.asarray(y_true, dtype=np.int64)
            scores_np = np.asarray(scores, dtype=np.float32)

            fpr, tpr, thr = roc_curve(y_true_np, scores_np, pos_label=1)
            fnr = 1.0 - tpr
            idx = int(np.nanargmin(np.abs(fpr - fnr)))
            eer = float((fpr[idx] + fnr[idx]) / 2.0)
            thresh = float(thr[idx])

            y_pred_spoof = scores_np >= thresh

            genuine = (y_true_np == 0)
            spoof   = (y_true_np == 1)

            correct_accept   = int(np.sum(genuine & ~y_pred_spoof))  # genuine przyjęte
            correct_reject   = int(np.sum(spoof   &  y_pred_spoof))  # spoof odrzucone
            wrong_reject     = int(np.sum(genuine &  y_pred_spoof))  # genuine odrzucone
            wrong_accept     = int(np.sum(spoof   & ~y_pred_spoof))  # spoof przyjęte

            self._log(
                f"[RESULT] EER = {eer*100:.2f}%  "
                f"(na {len(y_true_np)} próbkach, próg ≈ {thresh:.4f})"
            )
            self._log(
                "[STATS] Poprawnie zaakceptowano (GENUINE→ACCEPT): "
                f"{correct_accept}"
            )
            self._log(
                "[STATS] Poprawnie odrzucono (SPOOF→REJECT): "
                f"{correct_reject}"
            )
            self._log(
                "[STATS] Niepoprawnie odrzucono (GENUINE→REJECT): "
                f"{wrong_reject}"
            )
            self._log(
                "[STATS] Niepoprawnie zaakceptowano (SPOOF→ACCEPT): "
                f"{wrong_accept}"
            )


            # 4) Zapis wykresu ROC do pliku PNG obok pliku klucza
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            if labels_path_str:
                out_png = Path(labels_path_str).with_name(
                    f"roc_{mode}_{self.feat_var.get()}_{ts}.png"
                )
            else:
                out_png = eval_dir / f"roc_{mode}_{self.feat_var.get()}_{ts}.png"


            try:
                plt.figure()
                plt.plot(fpr, tpr, label=f"ROC (EER={eer*100:.2f}%)")
                plt.plot([0, 1], [0, 1], "--")
                plt.xlabel("False Positive Rate (FPR)")
                plt.ylabel("True Positive Rate (TPR)")
                plt.title("ROC – spoof vs genuine")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_png, dpi=150)
                plt.close()
                self._log(f"[INFO] Zapisano wykres ROC do: {out_png}")
            except Exception as e:
                self._log(f"[WARN] Nie udało się zapisać wykresu ROC: {e}")

        except Exception as e:
            self._error("Błąd ewaluacji EER", e)
        finally:
            self.btn_eval.configure(state="normal")

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 8}
        mode_frame = ttk.Frame(self)
        mode_frame.pack(fill="x", **pad)

        ttk.Label(mode_frame, text="Tryb:").grid(row=0, column=0, sticky="w")
        r1 = ttk.Radiobutton(
            mode_frame,
            text="Binary SVM (GENUINE/SPOOF)",
            variable=self.mode,
            value="binary",
            command=self._on_mode_change,
        )
        r2 = ttk.Radiobutton(
            mode_frame,
            text="One-Class (OC-SVM)",
            variable=self.mode,
            value="oneclass",
            command=self._on_mode_change,
        )
        r3 = ttk.Radiobutton(
            mode_frame,
            text="GMM (MFCC/LFCC/CQCC)",
            variable=self.mode,
            value="gmm",
            command=self._on_mode_change,
        )
        r1.grid(row=0, column=1, sticky="w", padx=(10, 0))
        r2.grid(row=0, column=2, sticky="w", padx=(12, 0))
        r3.grid(row=0, column=3, sticky="w", padx=(12, 0))

        models = ttk.LabelFrame(self, text="Modele / konfiguracja")
        models.pack(fill="x", **pad)
        self.models_frame = models

        self.bin_row = ttk.Frame(models)
        self.bin_row.grid(row=0, column=0, sticky="ew")
        ttk.Label(self.bin_row, text="Model SVM (.joblib):").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(self.bin_row, textvariable=self.bin_model_path).grid(
            row=0, column=1, sticky="w"
        )
        ttk.Button(
            self.bin_row, text="Wczytaj .joblib…", command=self.on_load_bin_model
        ).grid(row=0, column=2, padx=6)

        self.oc_row = ttk.Frame(models)
        self.oc_row.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(self.oc_row, text="Model one-class (.joblib):").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(self.oc_row, textvariable=self.oc_model_path).grid(
            row=0, column=1, sticky="w"
        )
        ttk.Button(self.oc_row, text="Wczytaj…", command=self.on_load_oc_model).grid(
            row=0, column=2, padx=6
        )

        self.gmm_row = ttk.Frame(models)
        self.gmm_row.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(self.gmm_row, text="Model GMM (.joblib):").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(self.gmm_row, textvariable=self.gmm_model_path).grid(
            row=0, column=1, sticky="w"
        )
        ttk.Button(
            self.gmm_row, text="Wczytaj .joblib…", command=self.on_load_gmm_model
        ).grid(row=0, column=2, padx=6)

        self.gmm_feat_row = ttk.Frame(models)
        self.gmm_feat_row.grid(row=4, column=0, sticky="ew", pady=(4, 0))
        ttk.Label(self.gmm_feat_row, text="Cecha dla GMM:").grid(
            row=0, column=0, sticky="w"
        )
        self.gmm_feat_combo = ttk.Combobox(
            self.gmm_feat_row,
            textvariable=self.gmm_feat_var,
            state="readonly",
            values=["lfcc", "mfcc", "cqcc"],
            width=10,
        )
        self.gmm_feat_combo.grid(row=0, column=1, sticky="w")
        ttk.Label(
            self.gmm_feat_row,
            text="(musi być zgodne z tym, jak trenowany był model GMM)",
        ).grid(row=0, column=2, sticky="w", padx=6)

        self.feat_row = ttk.Frame(models)
        self.feat_row.grid(row=5, column=0, sticky="ew", pady=(6, 2))
        ttk.Label(self.feat_row, text="Zestaw cech (--feat):").grid(
            row=0, column=0, sticky="w"
        )
        self.feat_combo = ttk.Combobox(
            self.feat_row,
            textvariable=self.feat_var,
            state="readonly",
            values=FEAT_CHOICES,
            width=18,
        )
        self.feat_combo.grid(row=0, column=1, sticky="w")
        ttk.Label(
            self.feat_row,
            text="(używane w trybie SVM/OC-SVM, musi być zgodne z treningiem)",
        ).grid(row=0, column=2, sticky="w", padx=6)

        audio_frame = ttk.LabelFrame(self, text="Audio do weryfikacji")
        audio_frame.pack(fill="x", **pad)
        ttk.Label(audio_frame, text="Plik audio:").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(audio_frame, textvariable=self.audio_path).grid(
            row=0, column=1, sticky="w"
        )
        ttk.Button(
            audio_frame, text="Wybierz plik .wav…", command=self.on_choose_audio
        ).grid(row=0, column=2, padx=6)

        eval_frame = ttk.LabelFrame(self, text="Testowanie")
        eval_frame.pack(fill="x", **pad)

        ttk.Label(eval_frame, text="Folder z plikami:").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(eval_frame, textvariable=self.eval_dir).grid(
            row=0, column=1, sticky="w"
        )
        ttk.Button(
            eval_frame,
            text="Wybierz folder…",
            command=self.on_choose_eval_dir,
        ).grid(row=0, column=2, padx=6)

        actions = ttk.Frame(self)
        actions.pack(fill="x", **pad)
        self.btn_predict = ttk.Button(
            actions, text="Predykcja", command=self.on_predict, state="disabled"
        )
        self.btn_predict.grid(row=0, column=0)
        self.btn_batch = ttk.Button(
            actions,
            text="Batch (folder → CSV)…",
            command=self.on_batch,
            state="disabled",
        )
        self.btn_batch.grid(row=0, column=1, padx=6)

        self.btn_eval = ttk.Button(
            actions,
            text="Ewaluacja EER…",
            command=self.on_eval_eer,
            state="disabled",
        )
        self.btn_eval.grid(row=0, column=2, padx=6)

        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=(4, 8))

        res = ttk.Frame(self)
        res.pack(fill="both", expand=True, **pad)
        ttk.Label(res, text="Wynik:").grid(row=0, column=0, sticky="w")
        self.lbl_result = ttk.Label(
            res, textvariable=self.result_var, font=("Segoe UI", 14, "bold")
        )
        self.lbl_result.grid(row=0, column=1, sticky="w", padx=8)

        ttk.Label(res, text="Log:").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.txt = tk.Text(res, height=14, wrap="word")
        self.txt.grid(row=2, column=0, columnspan=4, sticky="nsew")
        res.grid_columnconfigure(1, weight=1)
        res.grid_rowconfigure(2, weight=1)

        # Stopka
        bottom = ttk.Frame(self)
        bottom.pack(fill="x", **pad)
        ttk.Button(bottom, text="O programie", command=self.on_about).pack(
            side="left"
        )

        self._on_mode_change()

    def _on_mode_change(self) -> None:
        m = self.mode.get()
        if m == "binary":
            self.bin_row.grid()
            self.oc_row.grid_remove()
            self.gmm_row.grid_remove()
            self.gmm_feat_row.grid_remove()
            self.feat_row.grid()  # tylko w trybie binarnym

        elif m == "oneclass":
            self.bin_row.grid_remove()
            self.oc_row.grid()
            self.oc_enroll_row.grid()
            self.gmm_row.grid_remove()
            self.gmm_feat_row.grid_remove()
            self.feat_row.grid()  # one-class też korzysta z --feat

        else:  # gmm
            self.bin_row.grid_remove()
            self.oc_row.grid_remove()
            self.gmm_row.grid()
            self.gmm_feat_row.grid()
            self.feat_row.grid_remove()

        self._update_buttons()

    def on_load_bin_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Wybierz model SVM (.joblib)",
            filetypes=[("Joblib model", "*.joblib"), ("Wszystkie pliki", "*.*")],
        )
        if not path:
            return
        try:
            self.bin_model = joblib.load(path)
            self._bin_model_full = str(Path(path).resolve())
            self.bin_model_path.set(Path(path).name)
            self._log(f"Załadowano model SVM: {path}")
            self._update_buttons()
        except Exception as e:
            self._error("Błąd ładowania modelu SVM", e)

    def on_load_oc_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Wybierz model one-class (.joblib)",
            filetypes=[("Joblib model", "*.joblib"), ("Wszystkie pliki", "*.*")],
        )
        if not path:
            return
        try:
            self._oc_model_full = str(Path(path).resolve())
            self.oc_model_path.set(Path(path).name)
            self._log(f"Wybrano model one-class: {path}")
            self._update_buttons()
        except Exception as e:
            self._error("Błąd wyboru modelu one-class", e)

    def on_choose_enroll_dir(self) -> None:
        path = filedialog.askdirectory(title="Wybierz folder próbek enroll")
        if not path:
            return
        p = Path(path)
        self._oc_enroll_dir_full = str(p.resolve())
        self.oc_enroll_dir.set(str(p.resolve()))
        self._log(f"Wybrano folder enroll: {p}")

    def on_load_gmm_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Wybierz model GMM (.joblib)",
            filetypes=[("Joblib model", "*.joblib"), ("Wszystkie pliki", "*.*")],
        )
        if not path:
            return
        try:
            self.gmm_model = joblib.load(path)
            self._gmm_model_full = str(Path(path).resolve())
            self.gmm_model_path.set(Path(path).name)
            self._log(f"Załadowano model GMM: {path}")
            self._update_buttons()
        except Exception as e:
            self._error("Błąd ładowania modelu GMM", e)

    def on_choose_audio(self) -> None:
        path = filedialog.askopenfilename(
            title="Wybierz plik audio (.wav / .flac; 16 kHz mono zalecane)",
            filetypes=[
                ("Audio", "*.wav *.flac"),
                ("WAV", "*.wav"),
                ("FLAC", "*.flac"),
                ("Wszystkie pliki", "*.*"),
            ],
        )
        if not path:
            return
        p = Path(path)
        if p.suffix.lower() not in AUDIO_EXTS:
            messagebox.showwarning(
                "Uwaga",
                "Obsługiwane formaty to .wav oraz .flac (zalecane: 16 kHz, mono).",
            )
            return
        self.audio_path.set(str(p.resolve()))
        self._log(f"Wybrano plik audio: {p}")
        self._update_buttons()


    def on_predict(self) -> None:
        m = self.mode.get()
        audio_ok = self.audio_path.get() not in ("", "(nie wybrano)")
        if not audio_ok:
            return

        if m == "binary" and not self.bin_model:
            return
        if m == "oneclass" and not self._oc_model_full:
            return
        if m == "gmm" and not self.gmm_model:
            return

        self.btn_predict.configure(state="disabled")
        self.result_var.set("Predykcja w trakcie...")
        threading.Thread(target=self._predict_worker, daemon=True).start()

    def _predict_worker(self) -> None:
        try:
            wav = Path(self.audio_path.get())
            m = self.mode.get()

            if m == "binary":
                feats = extract_features_from_wav(
                    wav,
                    feature_types=self.feat_var.get(),
                ).reshape(1, -1)
                model = self.bin_model
                pred = int(model.predict(feats)[0])
                label = "SPOOF" if pred == 1 else "GENUINE"
                proba_txt = ""
                if hasattr(model, "predict_proba"):
                    try:
                        p = float(model.predict_proba(feats)[0, pred])
                        proba_txt = f" (p={p:.3f})"
                    except Exception:
                        pass
                self._set_result(f"{label}{proba_txt}")
                self._log(f"[BIN] {wav.name} -> {label}{proba_txt}")

            elif m == "oneclass":
                model, meta = load_enroll(self._oc_model_full)
                score, perc = predict_percent_notspoof(model, meta, wav)
                threshold = self.oc_perc_threshold

                if abs(score) >= threshold:
                    label = "SPOOF"
                else:
                    label = "GENUINE"

                txt = (
                    f"{label}  (SPOOF ≈ {perc:.1f}% )"
                )
                self._set_result(txt)
                self._log(f"[OC ] {wav.name} -> {txt}")
            else:  # gmm
                if self.gmm_model is None and self._gmm_model_full:
                    self.gmm_model = joblib.load(self._gmm_model_full)
                clf = self.gmm_model
                feat_type = self.gmm_feat_var.get()
                if feat_type == "mfcc":
                    x = mfcc_stats_feat(wav).reshape(1, -1)
                elif feat_type == "cqcc":
                    x = cqcc_stats_feat(wav).reshape(1, -1)
                else:  # lfcc
                    x = lfcc_stats_feat(wav).reshape(1, -1)
                yhat = int(clf.predict(x)[0])
                llr = float(clf.decision_function(x)[0])
                label = "SPOOF" if yhat == 1 else "GENUINE"
                txt = (
                    f"{label}  (LLR={llr:.3f}; "
                    f"LLR > 0 → GENUINE, LLR < 0 → SPOOF)"
                )
                self._set_result(txt)
                self._log(f"[GMM] {wav.name} -> {txt}")

        except Exception as e:
            self._error("Błąd predykcji", e)
        finally:
            self.btn_predict.configure(state="normal")


    def on_batch(self) -> None:
        m = self.mode.get()
        if m == "binary" and not self.bin_model:
            messagebox.showinfo("Info", "Najpierw wczytaj model SVM.")
            return
        if m == "oneclass" and not self._oc_model_full:
            messagebox.showinfo(
                "Info", "Najpierw wczytaj model one-class."
            )
            return
        if m == "gmm" and not self.gmm_model:
            messagebox.showinfo("Info", "Najpierw wczytaj model GMM.")
            return

        folder = filedialog.askdirectory(title="Wybierz folder z plikami audio")
        if not folder:
            return
        self._last_batch_dir = str(Path(folder).resolve())
        out_csv = filedialog.asksaveasfilename(
            title="Zapisz wyniki do CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Wszystkie pliki", "*.*")],
        )
        if not out_csv:
            return

        self.btn_batch.configure(state="disabled")
        threading.Thread(
            target=self._batch_worker, args=(Path(folder), Path(out_csv)), daemon=True
        ).start()

    def _batch_worker(self, folder: Path, out_csv: Path) -> None:
        try:
            files = []
            for pattern in ("*.wav", "*.flac"):
                files.extend(p for p in folder.rglob(pattern) if p.is_file())
            files = sorted(set(files))
            if not files:
                self._log("Brak plików audio w tym folderze.")
                return

            self._log(f"Batch: {len(files)} plików…")
            m = self.mode.get()

            with out_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)

                if m == "binary":
                    w.writerow(["path", "pred", "label_text", "proba"])
                    for i, p in enumerate(files, 1):
                        try:
                            feats = extract_features_from_wav(
                                p,
                                feature_types=self.feat_var.get(),
                            ).reshape(1, -1)
                            pred = int(self.bin_model.predict(feats)[0])
                            label = "SPOOF" if pred == 1 else "GENUINE"
                            proba = ""
                            if hasattr(self.bin_model, "predict_proba"):
                                try:
                                    proba = float(
                                        self.bin_model.predict_proba(feats)[0, pred]
                                    )
                                except Exception:
                                    proba = ""
                            w.writerow([str(p), pred, label, proba])
                        except Exception as e:
                            w.writerow([str(p), "ERR", str(e), ""])
                        if i % 10 == 0 or i == len(files):
                            self._log(f"… {i}/{len(files)}")

                elif m == "oneclass":
                    w.writerow(["path", "not_spoof_%", "score"])
                    model, meta = load_enroll(self._oc_model_full)
                    for i, p in enumerate(files, 1):
                        try:
                            score, perc = predict_percent_notspoof(model, meta, p)
                            w.writerow([str(p), f"{perc:.1f}", f"{score:.6f}"])
                        except Exception as e:
                            w.writerow([str(p), "ERR", str(e)])
                        if i % 10 == 0 or i == len(files):
                            self._log(f"… {i}/{len(files)}")

                else:  # gmm
                    clf = self.gmm_model
                    w.writerow(["path", "pred", "label_text", "llr"])
                    for i, p in enumerate(files, 1):
                        try:
                            feat_type = self.gmm_feat_var.get()
                            if feat_type == "mfcc":
                                x = mfcc_stats_feat(p).reshape(1, -1)
                            elif feat_type == "cqcc":
                                x = cqcc_stats_feat(p).reshape(1, -1)
                            else:
                                x = lfcc_stats_feat(p).reshape(1, -1)

                            yhat = int(clf.predict(x)[0])
                            llr = float(clf.decision_function(x)[0])
                            label = "SPOOF" if yhat == 1 else "GENUINE"
                            w.writerow([str(p), yhat, label, f"{llr:.6f}"])
                        except Exception as e:
                            w.writerow([str(p), "ERR", str(e), ""])
                        if i % 10 == 0 or i == len(files):
                            self._log(f"… {i}/{len(files)}")

            self._log(f"Zakończono. Zapisano: {out_csv}")
            messagebox.showinfo("Batch", f"Gotowe!\nZapisano: {out_csv}")
        except Exception as e:
            self._error("Błąd batch", e)
        finally:
            self.btn_batch.configure(state="normal")


    def _update_buttons(self) -> None:
        m = self.mode.get()
        audio_ok = self.audio_path.get() not in ("", "(nie wybrano)")

        eval_ready = self._eval_dir_full is not None

        if m == "binary":
            has_model = self.bin_model is not None
            can_pred = has_model and audio_ok
            can_batch = has_model
            can_eval = has_model and eval_ready

        elif m == "oneclass":
            has_oc = self._oc_model_full is not None
            can_pred = has_oc and audio_ok
            can_batch = has_oc
            can_eval = has_oc and eval_ready

        else:  # gmm
            has_gmm = self.gmm_model is not None
            can_pred = has_gmm and audio_ok
            can_batch = has_gmm
            can_eval = has_gmm and eval_ready

        self.btn_predict.configure(state="normal" if can_pred else "disabled")
        self.btn_batch.configure(state="normal" if can_batch else "disabled")
        self.btn_eval.configure(state="normal" if can_eval else "disabled")

    def _set_result(self, text: str) -> None:
        self.result_var.set(text)

    def _log(self, msg: str) -> None:
        self.txt.insert("end", msg + "\n")
        self.txt.see("end")

    def _error(self, title: str, e: Exception) -> None:
        self._log(f"[ERROR] {title}: {e}\n{traceback.format_exc()}".rstrip())
        messagebox.showerror(title, str(e))

    def _get_config_dict(self) -> dict:
        audio = self.audio_path.get()
        if audio in ("", "(nie wybrano)"):
            audio = None
        cfg = {
            "version": 1,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mode": self.mode.get(),
            "feat": self.feat_var.get(),
            "bin_model_path": self._bin_model_full,
            "oc_model_path": self._oc_model_full,
            "oc_enroll_dir": self._oc_enroll_dir_full,
            "gmm_model_path": self._gmm_model_full,
            "gmm_feat": self.gmm_feat_var.get(),
            "audio_path": audio,
            "last_batch_dir": self._last_batch_dir,
        }
        return cfg

    def _apply_config_dict(self, cfg: dict) -> None:
        mode = cfg.get("mode", "binary")
        if mode not in ("binary", "oneclass", "gmm"):
            mode = "binary"
        self.mode.set(mode)

        feat = cfg.get("feat")
        if isinstance(feat, str) and feat in FEAT_CHOICES:
            self.feat_var.set(feat)

        gmm_feat = cfg.get("gmm_feat")
        if isinstance(gmm_feat, str) and gmm_feat in ("lfcc", "mfcc", "cqcc"):
            self.gmm_feat_var.set(gmm_feat)

        bin_path = cfg.get("bin_model_path")
        if bin_path:
            p = Path(bin_path)
            if p.exists():
                try:
                    self.bin_model = joblib.load(str(p))
                    self._bin_model_full = str(p.resolve())
                    self.bin_model_path.set(p.name)
                    self._log(f"Załadowano model SVM z konfiguracji: {p}")
                except Exception as e:
                    self._error("Błąd ładowania SVM z konfiguracji", e)
            else:
                self._log(f"[WARN] Model SVM z konfiguracji nie istnieje: {p}")

        oc_model_path = cfg.get("oc_model_path")
        if oc_model_path:
            p = Path(oc_model_path)
            if p.exists():
                self._oc_model_full = str(p.resolve())
                self.oc_model_path.set(p.name)
                self._log(f"Ustawiono model one-class z konfiguracji: {p}")
            else:
                self._log(f"[WARN] Model one-class z konfiguracji nie istnieje: {p}")

        oc_enroll = cfg.get("oc_enroll_dir")
        if oc_enroll:
            p = Path(oc_enroll)
            if p.exists():
                self._oc_enroll_dir_full = str(p.resolve())
                self.oc_enroll_dir.set(str(p.resolve()))
                self._log(f"Ustawiono folder enroll z konfiguracji: {p}")
            else:
                self._log(f"[WARN] Folder enroll z konfiguracji nie istnieje: {p}")

        gmm_path = cfg.get("gmm_model_path")
        if gmm_path:
            p = Path(gmm_path)
            if p.exists():
                try:
                    self.gmm_model = joblib.load(str(p))
                    self._gmm_model_full = str(p.resolve())
                    self.gmm_model_path.set(p.name)
                    self._log(f"Załadowano GMM z konfiguracji: {p}")
                except Exception as e:
                    self._error("Błąd ładowania GMM z konfiguracji", e)
            else:
                self._log(f"[WARN] Model GMM z konfiguracji nie istnieje: {p}")

        audio_path = cfg.get("audio_path")
        if audio_path:
            p = Path(audio_path)
            if p.exists():
                self.audio_path.set(str(p.resolve()))
                self._log(f"Ustawiono plik audio z konfiguracji: {p}")
            else:
                self._log(f"[WARN] Plik audio z konfiguracji nie istnieje: {p}")

        last_batch = cfg.get("last_batch_dir")
        if last_batch:
            self._last_batch_dir = last_batch

        self._on_mode_change()

    def on_about(self) -> None:
        messagebox.showinfo(
            "AntiSpoof — GUI",
            "Interfejs dla trzech trybów pracy:\n"
            "• Binary SVM – klasyfikacja GENUINE/SPOOF, cechy MFCC/LFCC/CQCC "
            "(zestaw wybierany w polu 'Zestaw cech', musi być zgodny z treningiem).\n"
            "• One-Class (OC-SVM) – klasyfikator jednoklasowy trenowany na nagraniach "
            "autentycznych, korzysta z tych samych typów cech (--feat).\n"
            "• GMM – modele mieszanin Gaussa trenowane na MFCC/LFCC/CQCC.\n\n"
            "Tryb batch zapisuje wyniki do CSV, a ewaluacja EER pozwala policzyć EER "
            "i wykres ROC na bazie zadanego klucza etykiet.",
        )


def main() -> None:
    app = AntiSpoofGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
