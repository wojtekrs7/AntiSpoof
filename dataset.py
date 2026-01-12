from pathlib import Path
from typing import Iterator, Tuple

def iter_wavs_with_labels(root: Path) -> Iterator[tuple[Path, int]]:
    root = Path(root)
    for label, sub in [(0, "genuine"), (1, "spoof")]:
        d = root / sub
        if not d.is_dir():
            continue

        for p in d.rglob("*.wav"):
            yield p, label
