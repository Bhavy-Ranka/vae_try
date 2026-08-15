# trained seperate vae for each class
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

CSV_PATH = "cwru_features.csv"
FEATURE_COLS = ["mean", "RMS", "standard_deviation", "crest_factor", "skewness",
                 "shape_factor", "kurtosis", "peak_to_peak", "energy_factor",
                 "impulse_factor", "peak_frequency", "peak_to_peak_frequency",
                 "spectral_kurtosis", "spectral_bandwidth", "spectral_skewness"]
LABEL_COL = "label"

# 10% original data to train and 9 times more synthetic data than real data
SOURCE_SAMPLE_RATIO = 0.10
SYNTHETIC_TO_REAL_RATIO = 9

LATENT_DIM = 8
HIDDEN_DIM = 32
EPOCHS = 300
BATCH_SIZE = 32
LR = 1e-3
BETA = 0.5

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
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch = x_tensor[idx]
            opt.zero_grad()
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar, beta)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.shape[0]
        if (epoch + 1) % 50 == 0:
            print(f"epoch {epoch + 1}/{epochs} loss {total_loss / n:.4f}")
    return model


def generate_samples(model, n_samples, latent_dim, scaler):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim)
        x_gen = model.decode(z).numpy()
    return scaler.inverse_transform(x_gen)


def main():
    df = pd.read_csv(CSV_PATH)
    classes = sorted(df[LABEL_COL].unique())
    print(f"Classes found: {classes}")

    real_sampled_rows = []
    synthetic_rows = []

    for cls in classes:
        cls_df = df[df[LABEL_COL] == cls]
        n_available = len(cls_df)

        n_train = max(1, int(round(n_available * SOURCE_SAMPLE_RATIO)))
        n_gen = n_train * SYNTHETIC_TO_REAL_RATIO

        train_df = cls_df.sample(n=n_train, random_state=SEED)
        real_sampled_rows.append(train_df[[LABEL_COL] + FEATURE_COLS])

        scaler = StandardScaler()
        x_train = scaler.fit_transform(train_df[FEATURE_COLS].values)

        print(f"\n[Class {cls}] Available: {n_available} | 10% Real Source Train: {n_train} | 90% VAE Synthetic Gen: {n_gen}")
        model = train_vae_for_class(x_train, EPOCHS, BATCH_SIZE, LR, LATENT_DIM, HIDDEN_DIM, BETA)

        x_gen = generate_samples(model, n_gen, LATENT_DIM, scaler)
        gen_df = pd.DataFrame(x_gen, columns=FEATURE_COLS)
        gen_df[LABEL_COL] = cls
        synthetic_rows.append(gen_df)

    real_df = pd.concat(real_sampled_rows, ignore_index=True)
    synthetic_df = pd.concat(synthetic_rows, ignore_index=True)
    combined_df = pd.concat([real_df, synthetic_df], ignore_index=True)

    combined_out_name = "cwru_10real_90synthetic.csv"
    combined_df.to_csv(combined_out_name, index=False)

    synth_out_name = "cwru_vae_synthetic_only.csv"
    synthetic_df.to_csv(synth_out_name, index=False)
    print(f"Real source samples used (10%): {len(real_df)}")
    print(f"Synthetic VAE samples (90%):    {len(synthetic_df)}")
    print(f"Total combined rows:           {len(combined_df)}")
    print(f"Saved combined dataset to:     {combined_out_name}")
    print(f"Saved synthetic dataset to:    {synth_out_name}")

if __name__ == "__main__":
    main()