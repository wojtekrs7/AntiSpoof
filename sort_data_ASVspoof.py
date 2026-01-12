from pathlib import Path
import shutil


def parse_metadata_line(parts: list[str], track: str):
    if not parts:
        return None

    # znajdź label
    label = None
    for tok in parts:
        if tok in {"spoof", "bonafide"}:
            label = tok
            break
    if label is None:
        return None


    prefix = f"{track}_"
    utt_id = None
    for tok in parts:
        if tok.startswith(prefix):
            utt_id = tok
            break
    if utt_id is None:
        return None

    return utt_id, label


def prepare_split(track: str):
    ROOT = Path(f"data_asvspoof/ASVspoof2021_{track}_eval_part00/ASVspoof2021_{track}_eval")
    KEY = Path(f"data_asvspoof/{track}-keys-stage-1/keys/CM/trial_metadata.txt")
    OUT = Path(f"data_{track.lower()}_train")  # wyjściowy katalog dla CLI

    print(f"[INFO] Track={track}")
    print(f"[INFO] ROOT={ROOT}")
    print(f"[INFO] KEY={KEY}")
    print(f"[INFO] OUT={OUT}")

    if not ROOT.exists():
        raise FileNotFoundError(f"Brak katalogu audio: {ROOT}")

    if not KEY.exists():
        raise FileNotFoundError(f"Brak trial_metadata.txt: {KEY}")

    (OUT / "genuine").mkdir(parents=True, exist_ok=True)
    (OUT / "spoof").mkdir(parents=True, exist_ok=True)


    total_lines = 0
    total_valid = 0
    copied = 0
    cnt_genuine = 0
    cnt_spoof = 0

    PROGRESS_EVERY = 5000

    for line in KEY.open(encoding="utf-8"):
        total_lines += 1
        parts = line.strip().split()

        parsed = parse_metadata_line(parts, track)
        if parsed is None:
            continue

        utt_id, label = parsed
        total_valid += 1

        if total_valid % PROGRESS_EVERY == 0:
            print(
                f"[{track}] valid={total_valid}, copied={copied}, "
                f"genuine={cnt_genuine}, spoof={cnt_spoof}",
                flush=True,
            )

        src = ROOT / "flac" / f"{utt_id}.flac"
        if not src.exists():
            continue

        if label == "bonafide":
            cls = "genuine"
            cnt_genuine += 1
        else:
            cls = "spoof"
            cnt_spoof += 1

        dst = OUT / cls / f"{utt_id}.flac"
        shutil.copy2(src, dst)
        copied += 1

    print(
        f"[DONE] Track {track}: "
        f"linii={total_lines}, valid={total_valid}, "
        f"skopiowano={copied} (genuine={cnt_genuine}, spoof={cnt_spoof})"
    )


if __name__ == "__main__":
    for T in ["LA", "DF", "PA"]:
        prepare_split(T)
