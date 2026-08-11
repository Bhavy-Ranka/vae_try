# Ball fault (BF)
# Inner ring fault (IRF)
# Outer ring fault (ORF)
# Combination fault on both inner ring and outer ring (COM)
# Health woring state (NO)

import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

DATA_DIR = "seu_bearing"
OUTPUT_CSV = "seu_features.csv"
FS = 12000
WINDOW_SIZE = 2048
OVERLAP = 0.0
VIBRATION_COLUMN = 1     
PEEK_ONLY = False       

LABEL_MAP = {
    "health": "NO",
    "inner": "IRF",
    "outer": "ORF",
    "ball": "REF",
    "comb": "CMB",
}

def segment_signal(x, window_size, overlap=0.0):
    step = max(int(window_size * (1 - overlap)), 1)
    return [x[i:i + window_size] for i in range(0, len(x) - window_size + 1, step)]


def extract_features(x, fs):
    x = np.asarray(x, dtype=np.float64).ravel()
    L = len(x)

    mean_ = np.mean(x)
    rms = np.sqrt(np.mean(x ** 2))
    std_ = np.std(x)
    peak = np.max(np.abs(x))
    mean_abs = np.mean(np.abs(x))
    crest_factor = peak / rms if rms else 0.0
    skewness = skew(x, bias=True)
    shape_factor = rms / mean_abs if mean_abs else 0.0
    kurt = kurtosis(x, fisher=True, bias=True)
    p2p = np.max(x) - np.min(x)
    sum_sq = np.sum(x ** 2)
    sum_abs = np.sum(np.abs(x))
    energy_factor = sum_sq / (sum_abs ** 2) if sum_abs else 0.0
    impulse_factor = peak / mean_abs if mean_abs else 0.0

    fft_vals = np.fft.fft(x)
    half = L // 2
    mag = np.abs(fft_vals[:half])
    freqs = np.arange(half) * fs / L

    peak_idx = int(np.argmax(mag))
    peak_frequency = freqs[peak_idx]
    peak_to_peak_frequency = freqs[-1] - freqs[0]

    spectral_kurtosis = kurtosis(mag, fisher=True, bias=True)
    spectral_skewness = skew(mag, bias=True)

    mag_sum = np.sum(mag)
    mu_f = np.sum(freqs * mag) / mag_sum if mag_sum else 0.0
    spectral_bandwidth = (
        np.sqrt(np.sum(((freqs - mu_f) ** 2) * mag) / mag_sum) if mag_sum else 0.0
    )

    return {
        "mean": mean_, "RMS": rms, "standard_deviation": std_,
        "crest_factor": crest_factor, "skewness": skewness,
        "shape_factor": shape_factor, "kurtosis": kurt,
        "peak_to_peak": p2p, "energy_factor": energy_factor,
        "impulse_factor": impulse_factor,
        "peak_frequency": peak_frequency,
        "peak_to_peak_frequency": peak_to_peak_frequency,
        "spectral_kurtosis": spectral_kurtosis,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_skewness": spectral_skewness,
    }


def load_signal_column(path, col=VIBRATION_COLUMN):
    for kwargs in (
        dict(),
        dict(header=None),
        dict(header=None, sep=r"\s+"),
        dict(header=None, sep=r"\s+", skiprows=16), 
    ):
        try:
            df = pd.read_csv(path, **kwargs)
            if df.shape[1] > col:
                series = pd.to_numeric(df.iloc[:, col], errors="coerce").dropna()
                if len(series) > WINDOW_SIZE:
                    return series.to_numpy(dtype=np.float64)
        except Exception:
            continue
    raise ValueError(f"could not parse a usable numeric column {col} from {path}")


def peek_file(path, n=5):
    print(f"\n--- peek: {path} ---")
    for label, kwargs in (
        ("header guess", dict(nrows=n)),
        ("headerless guess", dict(header=None, nrows=n)),
        ("whitespace-sep headerless guess", dict(header=None, sep=r"\s+", nrows=n)),
    ):
        try:
            print(f"\n{label}:")
            print(pd.read_csv(path, **kwargs))
        except Exception as e:
            print(f"  failed: {e}")


def parse_filename(fname):
    base = os.path.basename(fname).replace(".csv", "")
    parts = base.split("_")  # e.g. ball_20_0 -> ['ball','20','0']
    fault_key = parts[0]
    label = LABEL_MAP.get(fault_key, "UNKNOWN")
    speed_hz = parts[1] if len(parts) > 1 else None
    load_v = parts[2] if len(parts) > 2 else None
    return label, speed_hz, load_v


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not files:
        print(f"No CSVs found in '{DATA_DIR}'. Fix DATA_DIR.")
        return

    if PEEK_ONLY:
        peek_file(files[0])
        print("\nSet PEEK_ONLY=False and confirm VIBRATION_COLUMN once the layout above looks right.")
        return

    rows = []
    for f in files:
        label, speed_hz, load_v = parse_filename(f)
        try:
            signal = load_signal_column(f)
        except Exception as e:
            print(f"  [skip] {f}: {e}")
            continue

        segments = segment_signal(signal, WINDOW_SIZE, OVERLAP)
        print(f"{os.path.basename(f)}: {len(signal)} samples -> {len(segments)} segments "
              f"[{label}, {speed_hz}Hz, {load_v}V]")

        for seg_id, seg in enumerate(segments):
            feats = extract_features(seg, FS)
            feats.update({
                "label": label,
                "speed_hz": speed_hz,
                "load_v": load_v,
                "segment_id": seg_id,
            })
            rows.append(feats)

    df = pd.DataFrame(rows)
    ordered_cols = ["label", "speed_hz", "load_v", "segment_id",
                     "mean", "RMS", "standard_deviation", "crest_factor", "skewness",
                     "shape_factor", "kurtosis", "peak_to_peak", "energy_factor",
                     "impulse_factor", "peak_frequency", "peak_to_peak_frequency",
                     "spectral_kurtosis", "spectral_bandwidth", "spectral_skewness"]
    df = df[ordered_cols]
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows x {len(ordered_cols)} cols to {OUTPUT_CSV}")
    print(df["label"].value_counts())

if __name__ == "__main__":
    main()