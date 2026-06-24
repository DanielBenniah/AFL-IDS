# AFL-IDS: Resource-Aware Federated Intrusion Detection for IoT Edge Computing

Reference implementation and reproducibility artifact for the paper:

> **AFL-IDS: Resource-Aware Federated Intrusion Detection for IoT Edge Computing**
> Daniel Benniah John, Department of Electrical Engineering and Computer Sciences, University of California, Berkeley
> Submitted to *PeerJ* (2026)

AFL-IDS is an Adaptive Federated Learning-based Intrusion Detection System that adds a **resource-aware client selection** mechanism to federated averaging. Instead of treating all clients uniformly, AFL-IDS selects only resource-sufficient IoT devices each round using a composite score:

```
R(i,t) = α·CPUᵢ(t) + β·MEMᵢ(t) + γ·BATᵢ(t),   α + β + γ = 1
```

A client participates in round *t* when `R(i,t) ≥ τ` (default `τ = 0.55`). If fewer than four clients meet the threshold, the four highest-scoring clients are selected to guarantee minimum participation. All raw traffic stays on the IoT devices; only model weights are exchanged.

> **Note on resources:** Resource scores are *simulated* to determine client eligibility only. Physical-device CPU, memory, battery, and latency are not measured, so no hardware savings are reported. This is a simulation study of the client-selection rule.

## Key Results

Across **5 independent trials** (seeds `{42, 123, 256, 789, 1024}`) on the TON_IoT Network Dataset:

| Method | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC-ROC (%) |
|---|---|---|---|---|---|
| **AFL-IDS (Ours)** | **98.06 ± 0.15** | **98.50 ± 0.15** | **97.26 ± 0.22** | **97.80 ± 0.13** | **99.20 ± 0.13** |
| Standard FedAvg | 88.96 ± 0.27 | 88.88 ± 0.49 | 88.11 ± 0.47 | 88.56 ± 0.36 | 93.98 ± 0.41 |
| FedProx | 90.28 ± 0.29 | 90.02 ± 0.31 | 89.48 ± 0.35 | 89.96 ± 0.32 | 95.04 ± 0.60 |
| Local-Only Training | 85.24 ± 0.65 | 85.13 ± 0.52 | 84.99 ± 0.42 | 84.96 ± 0.39 | 91.08 ± 0.40 |
| Centralised MLP | 92.89 ± 0.16 | 92.96 ± 0.24 | 92.63 ± 0.55 | 92.76 ± 0.28 | 96.43 ± 0.33 |

- AFL-IDS accuracy 95% CI `[97.87%, 98.24%]` does not overlap FedAvg's `[88.62%, 89.30%]` (paired *t*-test, `p < 0.01`).
- Converges to >97% accuracy by round 12; loss decreases from 0.4515 to 0.0870 over 15 rounds.
- Scales from 96.10% (5 clients) to 98.60% (50 clients) with std ≤ 0.21%.

## Method Overview

### Local detection model
A lightweight MLP (4,033 trainable parameters) suitable for constrained IoT devices without a GPU:

```
Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dropout(0.2) → Dense(1, sigmoid)
```

Optimised with Adam (`lr = 0.001`) and binary cross-entropy.

### Adaptive resource-aware FedAvg
Each round the edge server evaluates `R(i,t)` for all clients, selects those above `τ`, sends them the global weights, trains `E = 1` local epoch (batch size `B = 128`), and aggregates with weighted FedAvg:

```
wₜ₊₁ = Σ_{i∈S(t)} (nᵢ / Σ_{j∈S(t)} nⱼ) · wᵢᵗ⁺¹
```

Resource scores evolve as `R(i,t+1) = clip(R(i,t) + εᵢ, 0.3, 1.0)`, with `εᵢ ~ Uniform(-0.05, 0.05)` simulating dynamic IoT fluctuation.

### Preprocessing & reproducibility
- Train/test split is performed **before** any resampling or scaling.
- Class imbalance is handled on the **training partition only** with **SMOTENC** (categorical-aware), so synthetic samples respect nominal features. The test partition is never oversampled.
- `StandardScaler` is fit on the training partition only and applied unchanged to the test partition.
- Identifier columns are dropped; the `type` column is retained only to build non-IID partitions, never as a model input.
- Full determinism via `PYTHONHASHSEED=42`, `numpy.random.seed(42)`, `tf.random.set_seed(42)`, `TF_DETERMINISTIC_OPS=1`.

### Non-IID client partitioning
The 40,000 training samples are distributed across 10 clients by **attack-type skew**:

| Clients | Assigned traffic |
|---|---|
| 1–2 | DDoS, DoS |
| 3–4 | Backdoor, Injection |
| 5–6 | Password, Ransomware |
| 7–8 | Scanning, XSS |
| 9–10 | Normal, MitM |

Client sizes range from ~1,200 to ~6,800 samples (label distribution divergence δ² ≈ 0.18).

## Dataset

[**TON_IoT Network Dataset**](https://research.unsw.edu.au/projects/toniot-datasets) (Moustafa, 2021), created by the Cyber Range Lab of UNSW Canberra — 211,043 real network-traffic records across nine attack categories plus normal traffic. Per trial, 40,000 training and 10,000 test records are sampled.

This repository does **not** redistribute the dataset. Download `train_test_network.csv` from the link above and place it in the project root (or pass its path via `--data`).

## Installation

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

Core dependencies: `tensorflow`, `scikit-learn`, `imbalanced-learn` (for SMOTENC), `flwr` (Flower v1.30), `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`.

## Usage

```bash
python afl_ids_reproduce.py
```

This runs the full pipeline: preprocessing, the 5-trial federated experiment, baseline comparisons (FedAvg, FedProx, Local-Only, Centralised MLP), statistical validation (mean ± std, 95% CI, paired *t*-test), the ablation and sensitivity studies, and figure generation.

Optional flags:

```bash
python afl_ids_reproduce.py --data path/to/train_test_network.csv --out figures/
```

## Experimental Configuration

| Parameter | Value |
|---|---|
| Dataset | TON_IoT Network Dataset (211,043 records) |
| Features (preprocessed) | 29 |
| Train / Test per trial | 40,000 / 10,000 |
| Independent trials | 5 (seeds 42, 123, 256, 789, 1024) |
| FL clients (N) | 10 |
| FL rounds (T) | 15 |
| Local epochs (E) | 1 |
| Batch size (B) | 128 |
| Resource threshold (τ) | 0.55 |
| Optimizer | Adam (lr = 0.001) |
| Local model | MLP Dense(64)–Dense(32)–Dense(1), 4,033 params |
| FL framework | Flower (flwr) v1.30 |
| Confidence interval | 95% (Student's t, df = 4) |

## Generated Figures

Running the script produces the manuscript figures (300 DPI):

1. System architecture (three-tier IoT / Edge / Cloud)
2. Convergence — accuracy and loss over 15 rounds
3. Performance comparison with error bars
4. Confusion matrix (balanced n = 10,000 subset)
5. ROC curves (computed directly from saved labels and prediction probabilities)
6. Global-model loss across rounds
7. Scalability — accuracy vs. 5–50 clients

## Limitations

- Resource scores are simulated, not measured on hardware; results may differ on real devices subject to thermal throttling, OS scheduling, and network jitter.
- Binary (normal vs. attack) classification only; multi-class nine-category detection is future work.
- Adversarial robustness (model poisoning, gradient inversion, Byzantine failures) and formal privacy guarantees (differential privacy, secure aggregation) are not evaluated.
- FedNova, SCAFFOLD, and FedDyn are not compared due to higher communication overhead.

## Citation

```bibtex
@article{john2026aflids,
  title   = {AFL-IDS: Resource-Aware Federated Intrusion Detection for IoT Edge Computing},
  author  = {John, Daniel Benniah},
  journal = {PeerJ},
  year    = {2026}
}
```

## Acknowledgements

Thanks to the Cyber Range Lab of UNSW Canberra for making the TON_IoT dataset publicly available.

## License

The TON_IoT dataset is subject to its own license from UNSW Canberra. See the [dataset page](https://research.unsw.edu.au/projects/toniot-datasets) for terms.
