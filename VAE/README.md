# VAE Synthetic Data Quality Ratio Results

## 1. What this experiment measures

For each dataset (CWRU, Ottawa, SEU) and each real-data fraction (5%, 10%, 20%), the pipeline (`{dataset}_vae.py`) does the following:

1. Splits the full feature CSV into a **90/10 real train/test split**.
2. Trains a **CNN once** on the 90% real training split (this CNN is then frozen for the rest of the run).
3. For each fault class, samples `SOURCE_SAMPLE_RATIO` (5%/10%/20%) of that class's real rows and trains a **separate per-class VAE** (1 hidden layer, 32 units, 8-dim latent, MSE reconstruction + β·KL with β=0.5, 300 epochs) on just that subset.
4. At each checkpoint ratio (5x, 10x, 15x, 20x, 30x, 40x, 50x, 75x, 100x, 250x, [500x, 1000x for CWRU/SEU]), generates `n_train × ratio` synthetic rows per class from the **frozen** VAE.
5. Evaluates the **frozen, real-data-trained CNN** on this synthetic data and records accuracy.

**Critical framing:** this accuracy is *not* "does synthetic data help train a better classifier" (that would require training a new CNN on synthetic + real and testing on held-out real data). It is a **fidelity proxy**: does the CNN  which has only ever seen real data  still recognize the synthetic samples as belonging to the class the VAE intended? High accuracy means the VAE's outputs sit inside the real-data class boundaries the CNN learned; low accuracy means the VAE is drifting into feature-space regions the CNN doesn't associate with that class.

Feature set (15 dims, all runs): mean, RMS, standard deviation, crest factor, skewness, shape factor, kurtosis, peak-to-peak, energy factor, impulse factor, peak frequency, peak-to-peak frequency, spectral kurtosis, spectral bandwidth, spectral skewness.

---

## 2. Real-data CNN baseline

Before looking at synthetic data at all, this is how well the CNN classifies **held-out real data** (the 10% test split), trained on the 90% real split  this is the frozen classifier used to score all synthetic samples in section 3.

| Dataset | Baseline accuracy (real, 90/10 split) | Notes |
|---|---|---|
| CWRU | 99.49% | High fault detection across all fault types |
| Ottawa | 99.51% | Near-perfect across conditions |
| SEU | 91.78% | Good performance despite dataset complexity |

---

## 3. Results

### 3.1 CWRU

| Ratio | 5% real  acc | 10% real  acc | 20% real  acc |
|---|---|---|---|
| 5x | 97.67% | 96.01% | 97.47% |
| 10x | 98.22% | 96.49% | 97.23% |
| 15x | 98.22% | 97.16% | 97.18% |
| 20x | 98.22% | 97.06% | 97.35% |
| 30x | 97.99% | 97.41% | 97.24% |
| 40x | 97.53% | 96.98% | 97.26% |
| 50x | 97.78% | 96.93% | 97.15% |
| 75x | 97.82% | 97.16% | 97.25% |
| 100x | 98.01% | 96.93% | 97.26% |
| 250x | 97.88% | 96.91% | 97.20% |
| 500x | 97.94% | 96.89% | 97.28% |
| 1000x | 97.91% | 96.80% | 97.28% |

### 3.2 Ottawa

| Ratio | 5% real  acc | 10% real  acc | 20% real  acc |
|---|---|---|---|
| 5x | 92.69% | 94.07% | 95.66% |
| 10x | 93.31% | 93.95% | 95.16% |
| 15x | 92.52% | 94.30% | 95.50% |
| 20x | 92.78% | 94.39% | 95.57% |
| 30x | 92.84% | 94.20% | 95.49% |
| 40x | 92.68% | 94.30% | 95.46% |
| 50x | 92.41% | 94.40% | 95.48% |
| 75x | 92.40% | 94.34% | 95.53% |
| 100x | 92.37% | 94.21% | 95.47% |
| 250x | 92.51% | 94.25% | 95.51% |

### 3.3 SEU

| Ratio | 5% real  acc | 10% real  acc | 20% real  acc |
|---|---|---|---|
| 5x | 73.65% | 71.14% | 73.55% |
| 10x | 74.43% | 72.02% | 72.93% |
| 15x | 74.12% | 70.55% | 73.56% |
| 20x | 73.57% | 70.49% | 73.72% |
| 30x | 74.24% | 71.50% | 73.31% |
| 40x | 73.86% | 70.88% | 72.98% |
| 50x | 74.41% | 71.00% | 72.99% |
| 75x | 74.36% | 70.56% | 73.02% |
| 100x | 74.36% | 71.08% | 73.06% |
| 250x | 74.38% | 70.89% | 73.10% |
| 500x | 74.29% | 71.05% | 73.12% |
| 1000x | 74.35% | 70.87% | 73.18% |

### 3.4 Single shared VAE (CVAE)  20% real, one model for all classes

Same architecture (1 hidden layer, 32 units, 8-dim latent, β=0.5, 300 epochs) but trained **once** as a class-conditional VAE on the pooled 20%-real subset across all classes, instead of a separate VAE per class. Class identity is passed in as a one-hot conditioning vector on both encoder and decoder input. The frozen real-data CNN from section 2 is reused unchanged as the fidelity scorer.

| Ratio | CWRU  acc | Ottawa  acc | SEU  acc |
|---|---|---|---|
| 5x | 95.00% | 92.19% | 83.14% |
| 10x | 95.03% | 91.65% | 82.68% |
| 15x | 94.57% | 92.05% | 82.23% |
| 20x | 94.79% | 92.10% | 82.31% |
| 30x | 94.78% | 91.84% | 82.28% |
| 40x | 94.52% | 91.82% | 81.81% |
| 50x | 94.74% | 91.88% | 82.05% |
| 75x | 94.72% | 91.95% | 81.97% |
| 100x | 94.89% | 91.90% | 82.01% |
| 250x | 94.88% | 91.96% | 81.99% |
| 500x | 94.77% | 91.97% | 82.15% |
| 1000x | 94.79% | 91.93% | 82.10% |

The same two-phase shape (volatile 5x–50x, flat plateau ≥100x) shows up here too, confirming the sampling-variance explanation from §4.1 isn't specific to per-class VAEs  it's a property of small generation counts in general.

**Per-class VAE vs. single shared CVAE, plateau region (≥100x), 20% real:**

| Dataset | Baseline | Per-class VAE plateau | Per-class gap | Single CVAE plateau | Single CVAE gap | Change |
|---|---|---|---|---|---|---|
| CWRU | 99.49% | 97.20–97.28% | ~2.2–2.3pp | 94.77–94.89% | ~4.6–4.7pp | **worse, ~2.4pp** |
| Ottawa | 99.51% | 95.47–95.51% | ~4.0pp | 91.90–91.97% | ~7.5–7.6pp | **worse, ~3.5pp** |
| SEU | 91.78% | 73.06–73.18% | ~18.6–18.7pp | 81.99–82.15% | ~9.6–9.8pp | **better, ~9pp (gap nearly halved)** |
