# VAE Synthetic Data Quality — Ratio Results

## 1. What this experiment measures

For each dataset (CWRU, Ottawa, SEU) and each real-data fraction (5%, 10%, 20%), the pipeline (`{dataset}_vae.py`) does the following:

1. Splits the full feature CSV into a **90/10 real train/test split**.
2. Trains a **CNN once** on the 90% real training split (this CNN is then frozen for the rest of the run).
3. For each fault class, samples `SOURCE_SAMPLE_RATIO` (5%/10%/20%) of that class's real rows and trains a **separate per-class VAE** (1 hidden layer, 32 units, 8-dim latent, MSE reconstruction + β·KL with β=0.5, 300 epochs) on just that subset.
4. At each checkpoint ratio (5x, 10x, 15x, 20x, 30x, 40x, 50x, 75x, 100x, 250x, [500x, 1000x for CWRU/SEU]), generates `n_train × ratio` synthetic rows per class from the **frozen** VAE.
5. Evaluates the **frozen, real-data-trained CNN** on this synthetic data and records accuracy.

**Critical framing:** this accuracy is *not* "does synthetic data help train a better classifier" (that would require training a new CNN on synthetic + real and testing on held-out real data). It is a **fidelity proxy**: does the CNN — which has only ever seen real data — still recognize the synthetic samples as belonging to the class the VAE intended? High accuracy means the VAE's outputs sit inside the real-data class boundaries the CNN learned; low accuracy means the VAE is drifting into feature-space regions the CNN doesn't associate with that class.

Feature set (15 dims, all runs): mean, RMS, standard deviation, crest factor, skewness, shape factor, kurtosis, peak-to-peak, energy factor, impulse factor, peak frequency, peak-to-peak frequency, spectral kurtosis, spectral bandwidth, spectral skewness.

---

## 2. Real-data CNN baseline

Before looking at synthetic data at all, this is how well the CNN classifies **held-out real data** (the 10% test split), trained on the 90% real split — this is the frozen classifier used to score all synthetic samples in section 3.

| Dataset | Baseline accuracy (real, 90/10 split) | Notes |
|---|---|---|
| CWRU | 99.49% | High fault detection across all fault types |
| Ottawa | 99.51% | Near-perfect across conditions |
| SEU | 91.78% | Good performance despite dataset complexity |

---

## 3. Results

### 3.1 CWRU

| Ratio | 5% real — acc | 10% real — acc | 20% real — acc |
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

| Ratio | 5% real — acc | 10% real — acc | 20% real — acc |
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

| Ratio | 5% real — acc | 10% real — acc | 20% real — acc |
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

---

## 4. Analysis

### 4.1 Every curve has the same two-phase shape

**Phase 1 (ratio 5x–50x): volatile spikes and dips.**
`n_gen = n_train × ratio`, and `n_train` is only tens to a few hundred rows per class at these fractions. At ratio 5x with 5% real data, that's often under 200 synthetic samples per class total — small enough that a handful of atypical VAE draws swing the measured accuracy by a full percentage point or more. This is sampling variance, not evidence that the VAE is "better" or "worse" at these specific ratios.

**Phase 2 (ratio ≥100x): flat plateau.**
Once tens of thousands of samples are being drawn from the same frozen VAE, the law of large numbers takes over and the curve converges to the VAE's true generation quality. Because the VAE itself is never retrained between checkpoints, more synthetic volume past this point doesn't improve anything — it only tightens the accuracy estimate around a fixed value. Generating 1000x synthetic data buys essentially nothing over 100x–250x in any of the 9 runs.

**Practical implication:** treat only the plateau region (≥100x, ideally ≥250x) as meaningful signal. Anything drawn from the 5x–50x region is dominated by noise.

### 4.2 Dataset ceiling: it's mostly VAE fidelity, not CNN capacity

Now that the real-data baselines are known, the right comparison isn't "how close is synthetic accuracy to 100%" — it's **how close is synthetic accuracy to that dataset's own baseline**. That gap isolates VAE fidelity loss from whatever ceiling the CNN itself imposes.

| Dataset | Baseline (real) | Plateau range (synthetic) | Approx. gap to baseline |
|---|---|---|---|
| CWRU | 99.49% | 96.8% – 98.2% | **~1.3 – 2.7 pp** |
| Ottawa | 99.51% | 92.4% – 95.7% | **~3.8 – 7.1 pp** |
| SEU | 91.78% | 70.5% – 74.4% | **~17.4 – 21.3 pp** |

This changes the earlier conclusion meaningfully. SEU's baseline (91.78%) is not dramatically worse than CWRU's or Ottawa's near-perfect scores — it's a respectable classifier. But the synthetic-data gap for SEU is **6–10x larger** than CWRU's and roughly **3x larger** than Ottawa's. That means SEU's poor synthetic-accuracy numbers are overwhelmingly a **VAE fidelity failure**, not a "the CNN itself struggles with SEU" story as originally hypothesized.

- **CWRU**: the VAE reproduces the real-data class structure almost perfectly (gap under 3pp at every real-data fraction). This tiny 1-hidden-layer VAE is basically sufficient for CWRU's 15-feature space.
- **Ottawa**: a moderate, real fidelity loss (4–7pp), improving as real-data fraction increases — consistent with a VAE that's slightly under-capacity or under-trained for this feature distribution, but recoverable with more real data.
- **SEU**: a large, fraction-independent fidelity loss (~17–21pp) that doesn't meaningfully shrink even at 20% real data. This is the signature of the VAE **structurally failing** to capture SEU's class-conditional distributions — most likely because SEU's compound gearbox+bearing fault signatures produce feature distributions with much more inter-class overlap than CWRU or Ottawa, which this small linear-hidden-layer VAE (8-dim latent, MSE loss) doesn't have the capacity to separate. More real training rows for the VAE doesn't fix this, which points to an architecture/capacity limitation rather than a data-scarcity one.
