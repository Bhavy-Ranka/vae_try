import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# CSV File paths
ORIGINAL_CSV = "cwru_features.csv"
HYBRID_CSV = "cwru_10real_90synthetic.csv"
SYNTHETIC_CSV = "cwru_vae_synthetic_only9.csv"

FEATURE_COLS = [
    "mean", "RMS", "standard_deviation",
    "crest_factor", "skewness", "shape_factor",
    "kurtosis", "peak_to_peak", "energy_factor",
    "impulse_factor", "peak_frequency",
    "peak_to_peak_frequency", "spectral_kurtosis",
    "spectral_bandwidth", "spectral_skewness"
]
LABEL_COL = "label"

def load_data_and_preprocess():
    # 1. Load Original Dataset
    df_orig = pd.read_csv(ORIGINAL_CSV)

    label_encoder = LabelEncoder()
    y_orig_encoded = label_encoder.fit_transform(df_orig[LABEL_COL])
    num_classes = len(label_encoder.classes_)
    print(f"Classes found in original data: {label_encoder.classes_}")

    X_orig = df_orig[FEATURE_COLS].values

    # Train / Test split on original dataset (90% Train, 10% Test)
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
        X_orig, y_orig_encoded, test_size=0.10, random_state=42, stratify=y_orig_encoded
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_orig)
    X_test_orig_scaled = scaler.transform(X_test_orig)

    df_hybrid = pd.read_csv(HYBRID_CSV)
    y_hybrid_encoded = label_encoder.transform(df_hybrid[LABEL_COL])
    X_hybrid_scaled = scaler.transform(df_hybrid[FEATURE_COLS].values)

    df_synth = pd.read_csv(SYNTHETIC_CSV)
    y_synth_encoded = label_encoder.transform(df_synth[LABEL_COL])
    X_synth_scaled = scaler.transform(df_synth[FEATURE_COLS].values)
    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_orig_reshaped = X_test_orig_scaled.reshape(X_test_orig_scaled.shape[0], X_test_orig_scaled.shape[1], 1)
    X_hybrid_reshaped = X_hybrid_scaled.reshape(X_hybrid_scaled.shape[0], X_hybrid_scaled.shape[1], 1)
    X_synth_reshaped = X_synth_scaled.reshape(X_synth_scaled.shape[0], X_synth_scaled.shape[1], 1)

    return (
        X_train_reshaped, y_train_orig,
        X_test_orig_reshaped, y_test_orig,
        X_hybrid_reshaped, y_hybrid_encoded,
        X_synth_reshaped, y_synth_encoded,
        label_encoder, num_classes
    )


def build_1d_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    tf.random.set_seed(42)
    np.random.seed(42)
    (
        X_train, y_train,
        X_test_orig, y_test_orig,
        X_hybrid, y_hybrid,
        X_synth, y_synth,
        label_encoder, num_classes
    ) = load_data_and_preprocess()

    print("\n--- Building and Training 1D CNN Model on Original Data ---")
    model = build_1d_cnn_model((len(FEATURE_COLS), 1), num_classes)
    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    print("\n EVALUATION RESULTS ")
    test_loss_orig, test_acc_orig = model.evaluate(X_test_orig, y_test_orig, verbose=0)
    print(f"1. Original Test Set ({len(y_test_orig)} rows):          Accuracy = {test_acc_orig * 100:.2f}% | Loss = {test_loss_orig:.4f}")
    test_loss_hybrid, test_acc_hybrid = model.evaluate(X_hybrid, y_hybrid, verbose=0)
    print(f"2. Hybrid Dataset ({len(y_hybrid)} rows):             Accuracy = {test_acc_hybrid * 100:.2f}% | Loss = {test_loss_hybrid:.4f}")

    # Evaluate on Pure Synthetic Dataset
    test_loss_synth, test_acc_synth = model.evaluate(X_synth, y_synth, verbose=0)
    print(f"3. Pure Synthetic Dataset ({len(y_synth)} rows):     Accuracy = {test_acc_synth * 100:.2f}% | Loss = {test_loss_synth:.4f}")

    print("\n[A] Hybrid Dataset (10% Real + 90% VAE Synthetic):")
    y_pred_hybrid = np.argmax(model.predict(X_hybrid, verbose=0), axis=1)
    print(classification_report(y_hybrid, y_pred_hybrid, target_names=label_encoder.classes_))

    print("\n[B] Pure VAE Synthetic Dataset:")
    y_pred_synth = np.argmax(model.predict(X_synth, verbose=0), axis=1)
    print(classification_report(y_synth, y_pred_synth, target_names=label_encoder.classes_))


if __name__ == "__main__":
    main()
