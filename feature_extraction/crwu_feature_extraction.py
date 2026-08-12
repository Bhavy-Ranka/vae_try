#  drive-end vibrations
#  Normal , IRF , ORF , REF
import os
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import skew, kurtosis

DATA_DIR = "cwru"
OUTPUT_CSV = "cwru_features.csv"
FS = 12000
WINDOW_SIZE = 2048
OVERLAP = 0.0  

LOAD_RPM_MAP = {0: 1797, 1: 1772, 2: 1750, 3: 1730}

def fault_label_from_filename(fname):
    base = os.path.basename(fname).replace(".mat", "")
    parts = base.split("_")
    code = parts[0]
    load = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None

    if code.startswith("Normal"):
        return "NO", None, load
    if code.startswith("IR"):
        return "IRF", code[2:], load
    if code.startswith("OR"):
        return "ORF", code[2:5], load
    if code.startswith("B"):
        return "REF", code[1:], load
    return "UNKNOWN", None, load


def get_signal(mat_dict):
    for key in mat_dict:
        if key.endswith("DE_time"):
            return mat_dict[key].ravel(), "DE"
    for key in mat_dict:
        if key.endswith("FE_time"):
            print("  [warn] no DE channel, falling back to FE")
            return mat_dict[key].ravel(), "FE"
    raise KeyError("no *_DE_time or *_FE_time key in this .mat file")


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


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.mat")))
    if not files:
        print(f"No .mat files found in '{DATA_DIR}'. Fix DATA_DIR.")
        return

    rows = []
    for f in files:
        fault_type, fault_size, load = fault_label_from_filename(f)
        mat = loadmat(f)
        try:
            signal, channel = get_signal(mat)
        except KeyError as e:
            print(f"  [skip] {os.path.basename(f)}: {e}")
            continue

        segments = segment_signal(signal, WINDOW_SIZE, OVERLAP)
        print(f"{os.path.basename(f)} ({channel}): {len(signal)} samples -> "
              f"{len(segments)} segments [{fault_type}, size={fault_size}, load={load}HP]")

        for seg_id, seg in enumerate(segments):
            feats = extract_features(seg, FS)
            feats.update({
                "label": fault_type,
                "fault_size_inch": fault_size,
                "load_hp": load,
                "rpm": LOAD_RPM_MAP.get(load),
                "segment_id": seg_id,
            })
            rows.append(feats)

    df = pd.DataFrame(rows)
    ordered_cols = ["label", "fault_size_inch", "load_hp", "rpm", "segment_id",
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
