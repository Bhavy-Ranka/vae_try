import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import rfft, rfftfreq

RAW_DATA_DIR = "seu_bearing"          
SAMPLING_RATE = 12000              
WINDOW_SIZE = 2048                 
WINDOW_OVERLAP = 0.0               
OUTPUT_CSV = "seu_features2.csv"

LABEL_MAP = {
    "health": "NO",
    "inner": "IRF",
    "outer": "ORF",
    "ball": "REF",
    "comb": "CMB",
}


def load_signal(filepath, col=1):
    """
    Robustly loads numeric vibration data from SEU dataset files,
    handling comma, tab, or whitespace delimited files with headers.
    """
    for kwargs in (
        dict(header=None, sep=",", skiprows=16),
        dict(header=None, sep="\t", skiprows=16),
        dict(header=None, sep=r"\s+", skiprows=16),
        dict(header=None, sep=","),
        dict(header=None, sep="\t"),
        dict(header=None),
    ):
        try:
            df = pd.read_csv(filepath, **kwargs)
            if df.shape[1] > col:
                series = pd.to_numeric(df.iloc[:, col], errors="coerce").dropna()
                if len(series) >= WINDOW_SIZE:
                    return series.to_numpy(dtype=np.float64)
            series = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
            if len(series) >= WINDOW_SIZE:
                return series.to_numpy(dtype=np.float64)
        except Exception:
            continue

    try:
        data = np.loadtxt(filepath, delimiter=",")
        if data.ndim > 1:
            data = data[:, 0]
        return data.astype(np.float64)
    except Exception as e:
        raise ValueError(f"Could not load signal from {filepath}: {e}")


def make_windows(signal, window_size, overlap):
    step = int(window_size * (1 - overlap))
    step = max(step, 1)
    windows = []
    for start in range(0, len(signal) - window_size + 1, step):
        windows.append(signal[start:start + window_size])
    return windows


def time_domain_features(x):
    mean_ = np.mean(x)
    std_ = np.std(x)
    rms = np.sqrt(np.mean(x ** 2))
    peak = np.max(np.abs(x))
    peak_to_peak = np.max(x) - np.min(x)
    abs_mean = np.mean(np.abs(x))
    sqrt_abs_mean = (np.mean(np.sqrt(np.abs(x)))) ** 2

    skewness = float(stats.skew(x))
    kurtosis_val = float(stats.kurtosis(x, fisher=True))

    crest_factor = peak / rms if rms != 0 else 0.0
    shape_factor = rms / abs_mean if abs_mean != 0 else 0.0
    impulse_factor = peak / abs_mean if abs_mean != 0 else 0.0
    clearance_factor = peak / sqrt_abs_mean if sqrt_abs_mean != 0 else 0.0
    margin_factor = clearance_factor

    energy = np.sum(x ** 2)

    return {"mean": mean_,"std": std_,"rms": rms,"peak": peak,"peak_to_peak": peak_to_peak,"skewness": skewness,"kurtosis": kurtosis_val,"crest_factor": crest_factor,"shape_factor": shape_factor,
        "impulse_factor": impulse_factor,"clearance_factor": clearance_factor,"margin_factor": margin_factor,"energy": energy,}

def frequency_domain_features(x, fs):
    n = len(x)
    freqs = rfftfreq(n, d=1 / fs)
    spectrum = np.abs(rfft(x))
    spectrum_power = spectrum ** 2

    total_power = np.sum(spectrum_power)
    if total_power == 0:
        total_power = 1e-12

    spectral_centroid = np.sum(freqs * spectrum_power) / total_power

    if len(spectrum) > 1:
        peak_freq_idx = np.argmax(spectrum[1:]) + 1
        peak_frequency = freqs[peak_freq_idx]
        peak_amplitude = spectrum[peak_freq_idx]
    else:
        peak_frequency = 0.0
        peak_amplitude = 0.0

    spectral_bandwidth = np.sqrt(
        np.sum(((freqs - spectral_centroid) ** 2) * spectrum_power) / total_power
    )

    denom_skew = (spectral_bandwidth ** 3)
    spectral_skewness = (
        (np.sum(((freqs - spectral_centroid) ** 3) * spectrum_power) / total_power / denom_skew)
        if denom_skew > 1e-12 else 0.0
    )

    denom_kurt = (spectral_bandwidth ** 4)
    spectral_kurtosis = (
        (np.sum(((freqs - spectral_centroid) ** 4) * spectrum_power) / total_power / denom_kurt)
        if denom_kurt > 1e-12 else 0.0
    )

    psd_norm = spectrum_power / total_power
    psd_norm = psd_norm[psd_norm > 0]
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm))

    rms_frequency = np.sqrt(np.sum((freqs ** 2) * spectrum_power) / total_power)

    return {
        "spectral_centroid": spectral_centroid,"peak_frequency": peak_frequency,"peak_amplitude": peak_amplitude,"spectral_bandwidth": spectral_bandwidth,
        "spectral_skewness": spectral_skewness,"spectral_kurtosis": spectral_kurtosis,"spectral_entropy": spectral_entropy,"rms_frequency": rms_frequency,
    }

def extract_features(x, fs):
    features = {}
    features.update(time_domain_features(x))
    features.update(frequency_domain_features(x, fs))
    return features

def parse_filename_label(fname):
    base = os.path.basename(fname).lower()
    prefix = base.split("_")[0]
    return LABEL_MAP.get(prefix, prefix.upper())

def main():
    rows = []
    if not os.path.isdir(RAW_DATA_DIR):
        raise FileNotFoundError(
            f"'{RAW_DATA_DIR}' not found. Point RAW_DATA_DIR at your raw-signal folder."
        )

    class_dirs = sorted(
        d for d in os.listdir(RAW_DATA_DIR)
        if os.path.isdir(os.path.join(RAW_DATA_DIR, d))
    )

    file_tuples = []
    if class_dirs:
        print(f"Classes found in subdirectories: {class_dirs}")
        for label in class_dirs:
            class_path = os.path.join(RAW_DATA_DIR, label)
            for f in sorted(os.listdir(class_path)):
                if f.endswith((".csv", ".txt")):
                    file_tuples.append((os.path.join(class_path, f), label))
    else:
        flat_files = sorted(
            f for f in os.listdir(RAW_DATA_DIR)
            if f.endswith((".csv", ".txt"))
        )
        print(f"Flat dataset found with {len(flat_files)} data file(s) in '{RAW_DATA_DIR}'")
        for f in flat_files:
            fpath = os.path.join(RAW_DATA_DIR, f)
            label = parse_filename_label(f)
            file_tuples.append((fpath, label))

    if not file_tuples:
        print(f"No signal files (.csv / .txt) found in '{RAW_DATA_DIR}'.")
        return

    for fpath, label in file_tuples:
        fname = os.path.basename(fpath)
        try:
            signal = load_signal(fpath)
        except Exception as e:
            print(f"  Skipping {fname}: failed to load ({e})")
            continue

        windows = make_windows(signal, WINDOW_SIZE, WINDOW_OVERLAP)
        for w_idx, window in enumerate(windows):
            feats = extract_features(window, SAMPLING_RATE)
            feats["label"] = label
            feats["source_file"] = fname
            feats["window_idx"] = w_idx
            rows.append(feats)

        print(f"  [{label}] {fname}: {len(signal)} samples -> {len(windows)} windows extracted")

    if not rows:
        print("No features extracted — check RAW_DATA_DIR / file format.")
        return

    df = pd.DataFrame(rows)
    meta_cols = ["label", "source_file", "window_idx"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + feature_cols]

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows x {len(feature_cols)} features to '{OUTPUT_CSV}'")
    print(f"Class distribution:\n{df['label'].value_counts()}")


if __name__ == "__main__":
    main()
