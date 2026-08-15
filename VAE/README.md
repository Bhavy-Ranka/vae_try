# VAE Synthetic Data Augmentation — CWRU Bearing Fault Features

## Purpose

Test how much of a low-data regime (10% of real feature rows per class) can be
compensated for by generating synthetic rows with a per-class VAE, and whether
a classifier trained only on real data can still recognize that synthetic data
as belonging to the correct class.

## Pipeline

```
cwru_features.csv
      |
      v
vae_csv_generator.py  -->  cwru_10real_90synthetic.csv   (10% real rows + 90% VAE synthetic rows)
              -->  cwru_vae_synthetic_only.csv    (pure VAE synthetic rows, no real rows)
      |
      v
vae_csv_test.py  -->  1D CNN trained on 90/10 split of the ORIGINAL cwru_features.csv,
                       then tested on: original 10% holdout, hybrid CSV, synthetic-only CSV
```

## Files

- **`vae_csv_generator.py`** — trains one VAE per class (NO, IRF, ORF, REF) on a
  fraction of that class's real rows, then samples new rows from the trained
  decoder. Outputs the two CSVs above.
- **`vae_csv_test.py`** — trains a 1D CNN on 90% of the original real data,
  holds out 10% real as baseline test set, then evaluates the same trained
  model on the hybrid and synthetic-only CSVs to see if it still recognizes
  VAE-generated rows as the correct class.

## Key config (edit manually, no auto-sweep)

In `vae_csv_generator.py`:
- `SOURCE_SAMPLE_RATIO` — fraction of each class's real rows used to train
  that class's VAE (currently 0.10)
- `SYNTHETIC_TO_REAL_RATIO` — how many synthetic rows to generate per real
  row used (currently 9, i.e. 9x)
- `LATENT_DIM`, `HIDDEN_DIM`, `EPOCHS`, `BATCH_SIZE`, `LR`, `BETA` — VAE
  architecture/training knobs

## Design choices

- **One VAE per class, not one conditional VAE.** Each class's VAE only
  ever sees that class's data, so there's no risk of it blending fault
  signatures across classes. Trade-off: no shared structure learned across
  classes, and 4x the training calls.
- **Scaler fit per class, per training subset.** `StandardScaler` is fit
  only on the sampled training rows for that class, not the full dataset,
  to avoid leaking distribution info from held-out rows into the VAE.
- **Real and synthetic CSVs share identical columns** — `label` + the 15
  numeric features only. Metadata columns from the original CSV
  (`fault_size_inch`, `load_hp`, `rpm`, `segment_id`) are dropped before
  concatenation, since the VAE never generates them. Keeping them would
  leave synthetic rows with `NaN`/blank values in those columns.

## Latest results

Evaluated with a 1D CNN trained on the original 90/10 real split:

| Test set | Rows | Accuracy | Weighted precision | Weighted F1 |
|---|---|---|---|---|
| Hybrid (10% real + 90% synthetic) | 2960 | 96.82% | 0.97 | 0.97 |
| Pure synthetic | 2664 | 96.51% | 0.97 | 0.97 |

Per-class on both hybrid and synthetic sets: NO is near-perfect (VAE
reproduces the healthy-signal distribution cleanly). ORF is the weak point —
precision 1.00 but recall ~0.91–0.92, meaning the model never falsely calls
something ORF, but misses ~8-9% of true ORF rows because the VAE's ORF
samples drift close enough to a neighboring class's boundary.

## Known caveat — partial data leakage in hybrid number

`vae_csv_generator.py` samples its "10% real" rows from the full `cwru_features.csv`
independently (`random_state=42`) of the CNN's 90/10 `train_test_split`.
Some of those "real" rows inside `cwru_10real_90synthetic.csv` may also be
in the CNN's training set, meaning the model could have already seen them
during training. This only affects the real slice of the hybrid CSV (~296 of
2960 rows), so the impact on the reported 96.82% is small but non-zero. The
pure-synthetic number (96.51%) is unaffected, since none of those rows exist
in the original CSV.

