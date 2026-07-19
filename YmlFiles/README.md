# Conda Environment Configurations

This directory contains Conda environment specifications used throughout the development of the QDFT framework and its associated benchmarking projects.

Each environment targets a different stage of the research workflow, ranging from standard Qiskit-based simulations to GPU-accelerated PennyLane implementations and noisy quantum circuit simulations.

These YAML files allow users to recreate the exact software environments used during development, ensuring reproducibility across systems.

---

# Directory Contents

```text
YmlFiles/
│
├── namrataa_backup.yml
├── pl_env.yml
└── Noisy_Env.yml
```

---

# Environment Descriptions

## `namrataa_backup.yml`

Primary development environment used for the hybrid **QDFT (VQE + DFT)** framework.

This environment contains the software stack required for the majority of the repository, including:

* Qiskit
* Qiskit Nature
* PySCF
* Qiskit Aer
* Scientific Python libraries
* Visualization and development utilities

### Recommended Use

Use this environment when working with:

* QDFT (VQE + DFT)
* DFT + FCI Embedding
* Active-space embedding workflows
* Spin Hamiltonian implementations
* Production VQE calculations

---

## `pl_env.yml`

GPU-enabled environment for **PennyLane** benchmarking and quantum embedding studies.

This environment includes the dependencies required for GPU-accelerated variational quantum simulations, including:

* PennyLane
* PennyLane Lightning GPU
* CUDA libraries
* CuPy
* NVIDIA cuQuantum libraries
* Scientific Python ecosystem

### Recommended Use

Use this environment when working with:

* PennyLane GPU implementations
* Quantum Embedding Benchmarking Framework
* UCCSD and qUCCSD benchmarking
* GPU performance studies
* Cross-platform PennyLane vs. Qiskit comparisons

---

## `Noisy_Env.yml`

Environment dedicated to noisy quantum simulation experiments.

This configuration contains the packages required to construct and evaluate quantum circuits under realistic hardware noise models.

Typical dependencies include:

* Qiskit Aer
* IBM Quantum runtime packages
* Noise-model simulation tools
* Scientific Python libraries

### Recommended Use

Use this environment when working with:

* Noisy quantum simulations
* Hardware noise models
* Error analysis
* Quantum circuit fidelity studies
* Backend noise emulation

---

# Creating an Environment

Create an environment from any of the provided YAML files using:

```bash
conda env create -f <environment_file>.yml
```

For example:

```bash
conda env create -f namrataa_backup.yml
```

or

```bash
conda env create -f pl_env.yml
```

---

# Activating an Environment

After installation:

```bash
conda activate <environment_name>
```

For example:

```bash
conda activate Qiskit-1.0.2
```

or

```bash
conda activate pl_gpu_env
```

depending on the environment name specified within the YAML file.

---

# Updating an Existing Environment

If the environment already exists, update it using:

```bash
conda env update --file <environment_file>.yml --prune
```

---

# Environment Selection Guide

| Workflow                    | Recommended Environment  |
| --------------------------- | ------------------------ |
| QDFT (VQE + DFT)            | `namrataa_backup.yml`    |
| DFT + FCI Embedding         | `namrataa_backup.yml`    |
| PennyLane GPU Benchmarking  | `pl_env.yml`             |
| GPU-accelerated VQE studies | `pl_env.yml`             |
| Noisy Quantum Simulations   | `Noisy_Env.yml`          |
| Hardware Noise Benchmarking | `Noisy_Env.yml`          |

---

# Notes

* The environments are maintained independently because different projects require different software stacks and dependency versions.
* GPU acceleration requires a compatible NVIDIA GPU, appropriate CUDA drivers, and CUDA toolkit installation.
* Environment names may be modified after creation if desired, but preserving the provided configurations is recommended for reproducibility.
* When contributing to the repository, use the environment that corresponds to the project or directory being modified to avoid dependency conflicts.
