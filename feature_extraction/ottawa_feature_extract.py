# healthy , IRF, ORF, Ball Fault(ref) , cage fault(cf) 
import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

DATA_DIR = "ottawa_bearing"
OUTPUT_CSV = "ottawa_features.csv"
FS = 42000
WINDOW_SIZE = 2048
OVERLAP = 0.0
VIBRATION_COLUMN = 0     
PEEK_ONLY = False        

FOLDER_LABEL_MAP = {
    "1_Healthy": "NO",
    "2_Inner_Race_Faults": "IRF",
    "3_Outer_Race_Faults": "ORF",
    "4_Ball_Faults": "REF",
    "5_Cage_Faults": "CF",
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
    df = pd.read_csv(path)
    first_col = df.iloc[:, col]
    pd.to_numeric(first_col.iloc[:5])
    return df.iloc[:, col].to_numpy(dtype=np.float64)

def peek_file(path, n=5):
    print(f"\n--- peek: {path} ---")
    df_h = pd.read_csv(path, nrows=n)
    print("with header guess:")
    print(df_h)
    df_nh = pd.read_csv(path, header=None, nrows=n)
    print("headerless guess:")
    print(df_nh)


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*", "*.csv")))
    if PEEK_ONLY:
        peek_file(files[0])
        print("\nSet PEEK_ONLY=False and confirm VIBRATION_COLUMN once the layout above looks right.")
        return

    rows = []
    for f in files:
        folder = os.path.basename(os.path.dirname(f))
        label = FOLDER_LABEL_MAP.get(folder, "UNKNOWN")
        base = os.path.basename(f).replace(".csv", "")
        parts = base.split("_")

        try:
            signal = load_signal_column(f)
        except Exception as e:
            print(f"  [skip] {f}: {e}")
            continue

        segments = segment_signal(signal, WINDOW_SIZE, OVERLAP)
        print(f"{f}: {len(signal)} samples -> {len(segments)} segments [{label}]")

        for seg_id, seg in enumerate(segments):
            feats = extract_features(seg, FS)
            feats.update({
                "label": label,
                "bearing_id": parts[1] if len(parts) > 1 else None,
                "rep": parts[2] if len(parts) > 2 else None,
                "segment_id": seg_id,
            })
            rows.append(feats)

    df = pd.DataFrame(rows)
    ordered_cols = ["label", "bearing_id", "rep", "segment_id",
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