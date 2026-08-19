# Baseline Models Performance on Original Datasets
This section presents the performance evaluation of baseline 1D Convolutional Neural Network (CNN) models trained on the original CWRU, Ottawa, and SEU bearing fault datasets.
## Methodology
For each dataset, a 1D CNN model was trained using the following standard configuration:
- **Input Layer**: Accepts feature vectors extracted from the time-domain and frequency-domain signals.
- **Architecture**:
    - Conv1D (32 filters, kernel size 3, ReLU activation)
    - MaxPooling1D (pool size 2)
    - Conv1D (64 filters, kernel size 3, ReLU activation)
    - Flatten layer
    - Dense layer (64 units, ReLU activation)
    - Dropout layer (0.3)
    - Output Dense layer with softmax activation
- **Optimization**:
    - Optimizer: Adam (learning rate = 0.001)
    - Loss: Sparse categorical cross-entropy
- **Training**: 30 epochs, batch size 32, 20% validation split
- **Evaluation**: Performed on a hold-out test set (20% of original data)
## Results
### 1. CWRU Dataset
**Dataset Description**: Case Western Reserve University dataset with simulated faults (inner race, outer race, ball, and gear faults).
| Metric | Value |
|--------|-------|
| **Accuracy** | **99.49%** |
| Total Samples | 1242 |
| Test Samples | 311 |
---
**Conclusion**: The baseline CNN achieves near-perfect performance on the CWRU dataset, demonstrating effective feature extraction and classification capabilities.
### 2. Ottawa Dataset
**Dataset Description**: Dataset collected from University of Ottawa, featuring various defect types and operating conditions.
| Metric | Value |
|--------|-------|
| **Accuracy** | **99.51%** |
| Total Samples | 605 |
| Test Samples | 151 |
---
**Conclusion**: Exceptional performance with 99.51% accuracy, indicating the model can effectively distinguish between different fault types under various operating conditions.
### 3. SEU Dataset (updated — new feature extraction)
**Dataset Description**: Dataset collected from an industrial test rig at Southeast University.


**Update note:** The SEU result below reflects a re-run of the CNN baseline after replacing the original 15-feature time/frequency-domain extraction with an improved feature set (`feature_extraction/seu2.py`). Architecture, optimizer, and training config are unchanged from the methodology above.
| Metric | Value |
|--------|-------|
| **Accuracy_old** | **91.78%** |
| **Accuracy_new** | **99.22%** |


---
**Conclusion**: The new feature set closes almost the entire gap to CWRU/Ottawa (99.22% vs. 91.78% previously) — a ~7.4pp jump using the identical CNN architecture. This indicates the original SEU underperformance was primarily a **feature-representation problem**, not a dataset-complexity or model-capacity limitation as originally assumed. The new features apparently capture SEU's class-discriminative structure far more effectively than the original 15-feature set.

## Summary
The baseline 1D CNN model demonstrates excellent performance across all three benchmark datasets, with SEU now matching CWRU/Ottawa after the feature extraction update:

| Dataset | Accuracy | Key Characteristics |
|---------|----------|---------------------|
| **CWRU** | 99.49% | High fault detection rates across all fault types |
| **Ottawa** | 99.51% | Near-perfect performance under various conditions |
| **SEU (old features)** | 91.78% | Good performance despite dataset complexity |
| **SEU (new features)** | **99.22%** | Near-parity with CWRU/Ottawa after feature-extraction update |

**Revised interpretation:** the earlier "SEU is inherently harder" framing (dataset complexity, class overlap) was tied to the original 15-feature extraction. With improved features, SEU's real-data classification performance is now comparable to CWRU/Ottawa, which reframes the entire downstream VAE/synthetic-data analysis: the large per-class VAE fidelity gap seen with the old features may have been substantially a downstream symptom of weak features rather than an intrinsic VAE or dataset limitation. This needs to be re-validated by re-running the VAE ratio-sweep experiments (per-class and single-CVAE) on the new SEU feature set before finalizing that conclusion.

These results establish a strong foundation for evaluating the impact of synthetic data augmentation in the subsequent phases of the study.
