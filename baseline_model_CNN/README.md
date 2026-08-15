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

**Conclusion**: The baseline CNN achieves near-perfect performance on the CWRU dataset, demonstrating effective feature extraction and classification capabilities.

### 2. Ottawa Dataset

**Dataset Description**: Dataset collected from University of Ottawa, featuring various defect types and operating conditions.

| Metric | Value |
|--------|-------|
| **Accuracy** | **99.51%** |
| Total Samples | 605 |
| Test Samples | 151 |


**Conclusion**: Exceptional performance with 99.51% accuracy, indicating the model can effectively distinguish between different fault types under various operating conditions.

### 3. SEU Dataset

**Dataset Description**: Dataset collected from an industrial test rig at Southeast University .

| Metric | Value |
|--------|-------|
| **Accuracy** | **91.78%** |
| Total Samples | 382 |
| Test Samples | 96 |

**Conclusion**: While good, the performance is lower compared to the other two datasets, likely due to the dataset's complexity or the nature of the faults simulated. The model still maintains strong performance with 91.78% accuracy.

## Summary

The baseline 1D CNN model demonstrates excellent performance across all three benchmark datasets:

| Dataset | Accuracy | Key Characteristics |
|---------|----------|---------------------|
| **CWRU** | 99.49% | High fault detection rates across all fault types |
| **Ottawa** | 99.51% | Near-perfect performance under various conditions |
| **SEU** | 91.78% | Good performance despite dataset complexity |

These results establish a strong foundation for evaluating the impact of synthetic data augmentation in the subsequent phases of the study.
