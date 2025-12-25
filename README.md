# XOR Classification: Classical vs Quantum Machine Learning

An experimental research project comparing **classical machine learning models**
with a **Variational Quantum Classifier (VQC)** on the XOR classification problem.

The XOR task serves as a minimal yet fundamental benchmark for studying
**non-linearity, model expressivity, and optimization behavior**
in both classical and quantum learning systems.

This repository is structured as a **reproducible research project**
with automated experiment execution, figure generation, and result logging.


---


## Models Studied

The following models are implemented and evaluated:

- **Linear classifier**  
  Baseline model, incapable of solving XOR due to linear separability limits.

- **Multilayer Perceptron (MLP)**  
  Single hidden layer network with configurable number of hidden units.

- **Variational Quantum Classifier (VQC)**  
  Parameterized quantum circuit with classical optimization.


---


## Experimental Setup

### Dataset

Synthetic XOR datasets are generated from **Gaussian clusters** with configurable:

- noise level (σ)
- number of samples per cluster
- train / test split
- fixed random seeds for reproducibility


---


### Quantum Model Parameters

For the VQC, the following parameters are explored:

- circuit depth (number of variational layers)
- encoding and rotation gates
- number of measurement shots:
  - expectation value (`shots=None`)
  - finite sampling (`128`, `1024`)
- classical optimizer settings

## Repository Structure

```text
xor-qml-vs-classic/
├─ requirements.txt        # Python dependencies for full reproducibility
├─ LICENSE.txt             # Project license (e.g. MIT)
├─ README.md               # Project overview, usage instructions, experiment description
│
├─ src/                    # Core library code (no experiments here)
│  ├─ classic/             # Classical ML implementations
│  │  ├─ __init__.py       # Module initializer
│  │  ├─ activations.py    # Activation functions (e.g. sigmoid)
│  │  ├─ data.py           # XOR dataset generation and train/test split
│  │  ├─ linear.py         # Linear classifier (training + inference)
│  │  ├─ metrics.py        # Evaluation metrics (accuracy, loss)
│  │  ├─ mlp.py            # Multilayer Perceptron (MLP) implementation
│  │  ├─ plots.py          # Visualization utilities (scatter, boundaries, curves)
│  │  └─ utils.py          # Helper functions (I/O, directories, CSV saving)
│  │
│  └─ quantum/             # Quantum ML implementations
│     ├─ __init__.py       # Module initializer
│     ├─ data.py           # XOR data handling for quantum experiments
│     ├─ preprocess.py     # Feature scaling and angle encoding
│     ├─ vqc.py            # Variational Quantum Classifier (ansatz, loss, inference)
│     └─ metrics.py        # Evaluation metrics for quantum models
│
└─ experiments/            # Experiment scripts and generated results
   ├─ classic/             # Classical model experiments
   │  ├─ run_classic_grid.py      # Large-scale sweeps (noise, size, hidden units)
   │  ├─ run_classic_required.py  # Minimal required experiments for the paper
   │  └─ results/                 # Generated figures and CSV tables
   │     └─ ...
   │
   └─ quantum/             # Quantum model experiments
      ├─ run_vqc.py        # VQC experiments (training, depth, shots, robustness)
      └─ results/          # Generated quantum figures and metrics
         └─ ...

```

## Authors
- Miras Seilkhan | seilkhan.miras6117@gmail.com
- Adilbek Taizhanov | adilbek300108@gmail.com

