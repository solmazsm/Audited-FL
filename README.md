# Audited-FL: Securing Decentralized Federated Learning

**ICDE 2026 Submission**  

**Audited-FL** is a decentralized federated learning framework that integrates blockchain-based auditing and robust aggregation to defend against strategic, reward-driven attacks. It introduces and mitigates a novel **Gaussian incentive attack**, where colluding clients manipulate tokenized reward systems by injecting imperceptible noise into model updates—evading traditional anomaly detection while still degrading fairness.

---

## Key Features

- **Tamper-Evident Model Auditing**: Uses Merkle tree hashing and Byzantine fault-tolerant (BFT) consensus to log model updates on a permissioned blockchain (SciChain).
- **Defense Against Gaussian Attacks**: Detects collusive clients who inject low-variance noise to unfairly maximize token rewards.
- **Scalable Implementation**: Supports MPI-based training across 50–500 clients using FedML, PyTorch, and MPI4py.
- **Support for Robust Aggregation Methods**: Compatible with FedAvg, MultiKrum, FedDecorr, and FEDIC.

---

##  Directory Structure
```
Audited-FL/
├── GaussChain/ # Gaussian attack simulation and aggregation logic
├── FEDIC/ # Robust aggregation baseline (ICLR'23)
├── FedCLS/ # Additional baselines and experiment scripts
├── run_scripts/ # Shell scripts for experiments
├── utils.py # Utility functions
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```


## Experimental Setup

We implement **Audited-FL** using the **FedML** framework for federated learning and **SciChain** for blockchain-based auditing. To validate its effectiveness and scalability, we evaluate the framework across a wide range of benchmark datasets:

- **Standard Benchmarks**:  
  - **MNIST**  
  - **Fashion-MNIST**  
  - **CIFAR-10**  
  - **SVHN**

- **Large-Scale and Imbalanced Benchmarks**:  
  - **TinyImageNet** (200 classes, 64×64 resolution)  
  - **ImageNet-LT** (long-tailed version with 1,000 categories)

These datasets collectively demonstrate the model’s robustness across both balanced and highly imbalanced scenarios, as well as small- and large-scale data environments.

We compare **Audited-FL** using the following aggregation strategies to ensure a comprehensive and fair evaluation:

- **FedAvg** (classical baseline)  
- **MultiKrum** (robust against outliers)  
- **FedDecorr** (decorrelation-based defense)  
- **FEDIC** (recent state-of-the-art robust method)

This diversified setup enables rigorous testing against both classical and recent robust federated learning algorithms, including scenarios with noisy or malicious clients.

```
chmod +x run_fedic_experiments.sh
./run_fedic_experiments.sh
```

