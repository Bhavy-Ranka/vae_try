import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POSSIBLE_CSV_PATHS = [
    os.path.join(SCRIPT_DIR, "seu_features.csv"),
    os.path.join(SCRIPT_DIR, "..", "seu_features.csv"),
    "seu_features.csv"
]

CSV_PATH = None
for p in POSSIBLE_CSV_PATHS:
    if os.path.exists(p):
        CSV_PATH = p
        break

FEATURE_COLS = ["mean", "RMS", "standard_deviation", "crest_factor", "skewness",
                 "shape_factor", "kurtosis", "peak_to_peak", "energy_factor",
                 "impulse_factor", "peak_frequency", "peak_to_peak_frequency",
                 "spectral_kurtosis", "spectral_bandwidth", "spectral_skewness"]
LABEL_COL = "label"

SOURCE_SAMPLE_RATIO = 0.20
CHECKPOINT_RATIOS = [5, 10, 15, 20, 30, 40, 50, 75, 100, 250, 500, 1000]

LATENT_DIM = 8
HIDDEN_DIM = 32
VAE_EPOCHS = 300
VAE_BATCH_SIZE = 32
VAE_LR = 1e-3
BETA = 0.5

CNN_EPOCHS = 30
CNN_BATCH_SIZE = 32
SEED = 42

class SingleCVAE(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        self.enc1 = nn.Linear(input_dim + num_classes, hidden_dim)
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.dec1 = nn.Linear(latent_dim + num_classes, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x, c):
        inputs = torch.cat([x, c], dim=1)
        h = torch.relu(self.enc1(inputs))
        return self.enc_mu(h), self.enc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        inputs = torch.cat([z, c], dim=1)
        h = torch.relu(self.dec1(inputs))
        return self.dec2(h)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta):
    recon = F.mse_loss(recon_x, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kld


def train_single_cvae(x_train, c_train_onehot, epochs, batch_size, lr, latent_dim, hidden_dim, beta, num_classes):
    torch.manual_seed(SEED)
    input_dim = x_train.shape[1]
    model = SingleCVAE(input_dim, num_classes, hidden_dim, latent_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    c_tensor = torch.tensor(c_train_onehot, dtype=torch.float32)
    n = x_tensor.shape[0]
    
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch_x = x_tensor[idx]
            batch_c = c_tensor[idx]
            
            opt.zero_grad()
            recon, mu, logvar = model(batch_x, batch_c)
            loss = vae_loss(recon, batch_x, mu, logvar, beta)
            loss.backward()
            opt.step()
    return model


def generate_samples_for_class(model, class_idx, num_classes, n_samples, latent_dim, scaler):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim)
        c_onehot = torch.zeros(n_samples, num_classes)
        c_onehot[:, class_idx] = 1.0
        x_gen = model.decode(z, c_onehot).numpy()
    return scaler.inverse_transform(x_gen)


def build_cnn(input_shape, num_classes):
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    print(f"Loading dataset from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    classes = sorted(df[LABEL_COL].unique())
    print(f"Classes found: {classes}")

    label_encoder = LabelEncoder()
    y_all_encoded = label_encoder.fit_transform(df[LABEL_COL])
    num_classes = len(label_encoder.classes_)

    X_all = df[FEATURE_COLS].values
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
        X_all, y_all_encoded, test_size=0.10, random_state=SEED, stratify=y_all_encoded
    )

    cnn_scaler = StandardScaler()
    X_train_scaled = cnn_scaler.fit_transform(X_train_orig)
    X_test_scaled = cnn_scaler.transform(X_test_orig)
    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

    print("\nTraining CNN once on original 90/10 real split")
    cnn = build_cnn((len(FEATURE_COLS), 1), num_classes)
    cnn.fit(X_train_reshaped, y_train_orig, epochs=CNN_EPOCHS, batch_size=CNN_BATCH_SIZE,
            validation_split=0.2, verbose=0)
    test_loss, test_acc = cnn.evaluate(X_test_reshaped, y_test_orig, verbose=0)
    print(f"CNN baseline on real 10% holdout: accuracy={test_acc * 100:.2f}%")

    sampled_indices = []
    class_n_train = {}
    for cls in classes:
        cls_df = df[df[LABEL_COL] == cls]
        n_available = len(cls_df)
        n_train = max(1, int(round(n_available * SOURCE_SAMPLE_RATIO)))
        class_n_train[cls] = n_train
        sample_idx = cls_df.sample(n=n_train, random_state=SEED).index
        sampled_indices.extend(sample_idx)

    train_sample_df = df.loc[sampled_indices]
    X_train_vae_raw = train_sample_df[FEATURE_COLS].values
    y_train_vae_labels = train_sample_df[LABEL_COL].values
    y_train_vae_encoded = label_encoder.transform(y_train_vae_labels)

    vae_scaler = StandardScaler()
    X_train_vae_scaled = vae_scaler.fit_transform(X_train_vae_raw)

    y_train_vae_onehot = np.eye(num_classes)[y_train_vae_encoded]

    print(f"\nTraining SINGLE VAE (CVAE) on all {len(X_train_vae_scaled)} rows across {num_classes} classes...")
    single_vae_model = train_single_cvae(
        X_train_vae_scaled, y_train_vae_onehot,
        epochs=VAE_EPOCHS, batch_size=VAE_BATCH_SIZE, lr=VAE_LR,
        latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, beta=BETA,
        num_classes=num_classes
    )
    print("Single VAE training completed.")

    csv_output_dir = os.path.join(SCRIPT_DIR, "generated_csvs_1vae")
    os.makedirs(csv_output_dir, exist_ok=True)
    print(f"Saving generated CSV files to directory: {csv_output_dir}")

    results = []
    for ratio in CHECKPOINT_RATIOS:
        rows_list = []
        labels_list = []

        for c_idx, cls in enumerate(label_encoder.classes_):
            n_train = class_n_train[cls]
            n_gen = n_train * ratio

            gen_samples = generate_samples_for_class(
                single_vae_model, c_idx, num_classes, n_gen, LATENT_DIM, vae_scaler
            )
            rows_list.append(gen_samples)
            labels_list.extend([cls] * n_gen)

        X_synth = np.vstack(rows_list)
        y_synth_encoded = label_encoder.transform(np.array(labels_list))

        df_synth = pd.DataFrame(X_synth, columns=FEATURE_COLS)
        df_synth[LABEL_COL] = labels_list
        df_synth = df_synth[[LABEL_COL] + FEATURE_COLS]

        csv_file_path = os.path.join(csv_output_dir, f"seu_1vae_synthetic_ratio_{ratio}x.csv")
        df_synth.to_csv(csv_file_path, index=False)

        X_synth_scaled = cnn_scaler.transform(X_synth)
        X_synth_reshaped = X_synth_scaled.reshape(X_synth_scaled.shape[0], X_synth_scaled.shape[1], 1)

        loss, acc = cnn.evaluate(X_synth_reshaped, y_synth_encoded, verbose=0)
        print(f"ratio {ratio:4d}x -> {len(y_synth_encoded):6d} synthetic rows saved to '{os.path.basename(csv_file_path)}', accuracy={acc * 100:.2f}%")
        results.append({"ratio": ratio, "n_rows": len(y_synth_encoded), "accuracy": acc, "csv_file": csv_file_path})

    main_csv_path = os.path.join(SCRIPT_DIR, "seu_1vae_synthetic.csv")
    main_synth_df = pd.read_csv(os.path.join(csv_output_dir, "seu_1vae_synthetic_ratio_10x.csv"))
    main_synth_df.to_csv(main_csv_path, index=False)
    print(f"\nSaved representative synthetic CSV: {main_csv_path}")

    results_df = pd.DataFrame(results)
    # summary_csv_path = os.path.join(SCRIPT_DIR, "cwru_1vae_ratio_sweep_results.csv")
    # results_df.to_csv(summary_csv_path, index=False)
    # print(f"Saved sweep results summary: {summary_csv_path}")

    plot_path = os.path.join(SCRIPT_DIR, f"{int(SOURCE_SAMPLE_RATIO*100)}seu_1vae_ratio_sweep_plot.png")
    plt.figure(figsize=(7, 5))
    plt.plot(results_df["ratio"], results_df["accuracy"] * 100, marker="o", color="indigo", linewidth=2)
    plt.xlabel("Synthetic-to-real ratio")
    plt.ylabel("CNN accuracy on synthetic data (%)")
    plt.title("Single VAE (CVAE): Synthetic data quality vs generation ratio")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
