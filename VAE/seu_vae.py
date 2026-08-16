import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

CSV_PATH = "seu_features.csv"
FEATURE_COLS = ["mean", "RMS", "standard_deviation", "crest_factor", "skewness",
                 "shape_factor", "kurtosis", "peak_to_peak", "energy_factor",
                 "impulse_factor", "peak_frequency", "peak_to_peak_frequency",
                 "spectral_kurtosis", "spectral_bandwidth", "spectral_skewness"]
LABEL_COL = "label"

SOURCE_SAMPLE_RATIO = 0.20
CHECKPOINT_RATIOS = [5, 10, 15, 20, 30, 40, 50, 75, 100, 250]

LATENT_DIM = 8
HIDDEN_DIM = 32
VAE_EPOCHS = 300
VAE_BATCH_SIZE = 32
VAE_LR = 1e-3
BETA = 0.5

CNN_EPOCHS = 30
CNN_BATCH_SIZE = 32
SEED = 42

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.enc1 = nn.Linear(input_dim, hidden_dim)
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.dec1 = nn.Linear(latent_dim, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = torch.relu(self.enc1(x))
        return self.enc_mu(h), self.enc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.dec1(z))
        return self.dec2(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta):
    recon = nn.functional.mse_loss(recon_x, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kld


def train_vae_for_class(x_train, epochs, batch_size, lr, latent_dim, hidden_dim, beta):
    torch.manual_seed(SEED)
    model = VAE(x_train.shape[1], hidden_dim, latent_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    n = x_tensor.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch = x_tensor[idx]
            opt.zero_grad()
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar, beta)
            loss.backward()
            opt.step()
    return model


def generate_samples(model, n_samples, latent_dim, scaler):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim)
        x_gen = model.decode(z).numpy()
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

    class_vaes = {}
    for cls in classes:
        cls_df = df[df[LABEL_COL] == cls]
        n_available = len(cls_df)
        n_train = max(1, int(round(n_available * SOURCE_SAMPLE_RATIO)))
        train_df = cls_df.sample(n=n_train, random_state=SEED)

        vae_scaler = StandardScaler()
        x_train = vae_scaler.fit_transform(train_df[FEATURE_COLS].values)

        print(f"\n[Class {cls}] training VAE on {n_train} real rows")
        vae_model = train_vae_for_class(x_train, VAE_EPOCHS, VAE_BATCH_SIZE, VAE_LR,
                                         LATENT_DIM, HIDDEN_DIM, BETA)

        class_vaes[cls] = {"model": vae_model, "scaler": vae_scaler, "n_train": n_train}
        print(f"[Class {cls}] VAE trained and fixed, will generate directly per checkpoint")

    results = []
    for ratio in CHECKPOINT_RATIOS:
        rows = []
        labels = []
        for cls in classes:
            n_train = class_vaes[cls]["n_train"]
            vae_model = class_vaes[cls]["model"]
            vae_scaler = class_vaes[cls]["scaler"]
            n_gen = n_train * ratio
            gen = generate_samples(vae_model, n_gen, LATENT_DIM, vae_scaler)
            rows.append(gen)
            labels.extend([cls] * n_gen)

        X_synth = np.vstack(rows)
        y_synth = label_encoder.transform(np.array(labels))
        X_synth_scaled = cnn_scaler.transform(X_synth)
        X_synth_reshaped = X_synth_scaled.reshape(X_synth_scaled.shape[0], X_synth_scaled.shape[1], 1)

        loss, acc = cnn.evaluate(X_synth_reshaped, y_synth, verbose=0)
        print(f"ratio {ratio}x -> {len(y_synth)} synthetic rows, accuracy={acc * 100:.2f}%")
        results.append({"ratio": ratio, "n_rows": len(y_synth), "accuracy": acc})

    results_df = pd.DataFrame(results)
    # results_df.to_csv("vae_ratio_sweep_results.csv", index=False)
    # print("\nsaved vae_ratio_sweep_results.csv")

    plt.figure(figsize=(7, 5))
    plt.plot(results_df["ratio"], results_df["accuracy"] * 100, marker="o")
    plt.xlabel("Synthetic-to-real ratio")
    plt.ylabel("CNN accuracy on synthetic data (%)")
    plt.title("Synthetic data quality vs generation ratio (fixed VAE, fixed CNN)")
    plt.grid(True)
    plt.savefig(f"{SOURCE_SAMPLE_RATIO*100}ottawa_vae_ratio_sweep_plot.png", dpi=150, bbox_inches="tight")
    # print("saved vae_ratio_sweep_plot.png")

if __name__ == "__main__":
    main()
