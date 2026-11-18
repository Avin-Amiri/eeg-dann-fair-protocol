# EEG DANN Fair Protocol

Cross-subject EEG emotion recognition implementation on the SEED dataset using Domain Adversarial Neural Networks (DANN) and multi-source voting, with a strictly evaluated fair evaluation protocol that prevents data leakage between calibration/training and test sets.

---

## Overview

This repository contains experimental scripts and protocol implementations for subject-to-subject transfer learning on EEG emotion recognition tasks. The core focus is contrasting standard overlapping (leaky) evaluation baselines against a strict **Fair Evaluation Protocol** that enforces disjoint calibration and test splits with a temporal gap.

### Key Components

*   **`MainSeed.py` & `protocol_mine.py`:** Core training and evaluation pipeline supporting both `fair` and `base` protocols. Implements source-supervised training, domain discriminator updates, feature mixing, and pseudo-label adaptation.
*   **`main_seed_6_fair.py`:** A fair-only direct-training variant reporting last-epoch metrics alongside an oracle test upper bound.
*   **`Vote.py`:** Multi-source selection and voting pipeline. Ranks candidate source models using calibration loss, selects the top-performing sources (up to 5), and aggregates their predictions via majority vote.
*   **`Loading_Data.py`:** Parses subject MATLAB (`.mat`) files from the SEED dataset, extracting and separately normalizing DE+LDS features (frequency bands divided into two groups).
*   **`leak_check.py`:** A 1-Nearest-Neighbor diagnostic tool comparing labeled-test overlap and nearest-neighbor classification accuracy under base vs. fair splits.

---

## Evaluation Protocols

1.  **Fair Protocol (`--protocol fair`):**
    *   For each trial, the first $N$ windows (default: 12) serve as calibration/training data.
    *   The final 30% of windows serve as the test set.
    *   A temporal gap (default: 10 windows) is enforced between the labeled pool and the test set to ensure disjoint sets and prevent temporal leakage.
2.  **Base Protocol (`--protocol base`):**
    *   The initial windows are labeled for calibration, but the test set encompasses the entire target subject (including the calibration windows), resulting in a leaky comparison.

---

## Project Structure

```text
├── MainSeed.py             # Main training script (supports fair/base)
├── main_seed_6_fair.py     # Direct fair-only training script
├── Vote.py                 # Source selection and voting script
├── Loading_Data.py         # SEED dataset loader & feature extractor
├── protocol_mine.py        # Split definitions and pseudo-labeling logic
├── leak_check.py           # 1-NN leakage diagnostic tool
└── models/                 # Neural network architecture definitions
```

---

## Requirements & Dependencies

The codebase relies on standard scientific and deep learning packages:
*   Python 3.x
*   PyTorch
*   NumPy
*   SciPy

---

## Usage

Run the scripts from the repository root using Python module execution:

```bash
# Run the core protocol training (check script for default arguments)
python -m experiments.MainSeed --help

# Run multi-source voting evaluation
python -m experiments.Vote --help
```

---

## Reported Results

As documented in the project evaluation report, performance across sessions and protocols:

| Scenario / Protocol | Session | Calibration Samples | Single-Model Accuracy | Voting Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Base Protocol** | Session 3 | 180 | — | 89.23% |
| **Fair + Pseudo-Labeling** | Session 1 | 180 | 76.60% | 86.85% |
| **Fair + Pseudo-Labeling** | Session 2 | 180 | 75.76% | 84.58% |
| **Fair + Pseudo-Labeling** | Session 3 | 180 | 73.44% | 81.41% |
| **Direct Fair (Last Epoch)** | Session 3 | 180 | 68.83% | — |
| **Direct Fair (Oracle Upper Bound)** | Session 3 | 180 | 78.03% * | — |

*\*Note: The oracle upper bound metric uses test-accuracy-based model selection and represents an upper bound rather than an unbiased selected-model score.*

### Leakage Diagnostic
A 1-NN classifier diagnostic yields mean accuracies of **89.62% (Base)** versus **81.22% (Fair)**—an 8.4 percentage point drop when removing data leakage. *Caution:* Feature normalization in these experiments was applied to the entire target subject data prior to test splitting, which remains a factor to account for in strict zero-shot settings.

---

## References

*   **Reference Method:** Xiangyu Ju, Xu Wu, Sheng Dai, Ming Li, and Dewen Hu, "Domain adversarial learning with multiple adversarial tasks for EEG emotion recognition," *Expert Systems With Applications*, 266 (2025), 126028. DOI: [10.1016/j.eswa.2024.126028](https://doi.org/10.1016/j.eswa.2024.126028).
*   **Dataset:** SEED Dataset (SJTU BCMI Laboratory).
