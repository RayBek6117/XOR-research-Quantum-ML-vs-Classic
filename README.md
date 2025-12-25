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
├─ requirements.txt
├─ src/
│  └─ quantum/
│     ├─ __init__.py
│     ├─ data.py          # XOR data generation + split
│     ├─ preprocess.py    # scaling -> angles
│     ├─ vqc.py           # circuit builder + loss + predict
│     └─ metrics.py       # accuracy + helpers
└─ experiments/
   └─ quantum/
      └─ run_vqc.py       # single entry-point: trains + plots + saves CSV
```

## Authors
- Miras Seilkhan  
Email: seilkhan.miras6117@gmail.com
- Adilbek Taizhanov
Email: adilbek300108@gmail.com

