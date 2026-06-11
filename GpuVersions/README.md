Quantum Embedding Benchmarking Framework

This repository contains the progressive development, optimization, and benchmarking of a hybrid quantum-classical embedding simulation framework utilizing PennyLane and Qiskit. The project tracks the evolution of implementing Variational Quantum Eigensolver (VQE) workflows optimized for high-performance GPU acceleration, culminating in an architectural analysis of JIT compilation limits and a rigorous CPU vs. GPU performance comparison.

├── PennyLane Version 1/
│   └── PLgpu_v1.py [GPU Implementation]
├── PennyLane Version 2/
│   ├── UCCSD/
│   │   ├── PLgpu_v2_spin.py (with Active Space Spin Penalty Patch)
│   │   ├── PLcpu_v2_spin.py (with Active Space Spin Penalty Patch)
│   │   ├── Qkit_v2_spin.py  (with Active Space Spin Penalty Patch)
│   │   └── PLgpu_v2.py      (without Active Space Spin Penalty Patch)        
│   └── qUCCSD/
│       ├── PLgpu_qUCCSD_spin.py (with Active Space Spin Penalty Patch)
│       ├── PLcpu_qUCCSD_spin.py (with Active Space Spin Penalty Patch)
│       └── Qkit_qUCCSD_spin.py  (with Active Space Spin Penalty Patch)
└── PennyLane Version 3/
    └── PLgpu_v3.py [JAX-JIT GPU Implementation]

Architectural Evolution & Iterations

Version 1: Naive Hybrid Framework (Baseline GPU)
-> Ansatz: Built via Qiskit Nature (QiskitUCCSD).
-> Fermionic-to-Qubit Mapping: ParityMapper (reducing 2 qubits to optimize memory overhead).
-> Execution Bottleneck: Required explicit bound_circuit.decompose() and qml.from_qiskit() conversion steps inside the self-consistent optimization loop to translate the Qiskit circuit object into a native PennyLane format.
-> Gradient Profile: Relied on SciPy’s default numerical Finite Differences, resulting in massive computational overhead (150+ circuit evaluations per optimization step).

Version 2: Optimized Native Execution & Exact Gradients (Production Baseline)
-> Ansatz: Built natively via PennyLane (qml.UCCSD).
-> Fermionic-to-Qubit Mapping: Shifted to JordanWignerMapper to strictly satisfy PennyLane's template requirement of 1 spatial orbital to 2 qubits.
-> Execution Backend: Scaled directly via lightning.gpu with no internal object translation or compilation bottlenecks.
-> Gradient Profile: Integrated exact analytical math by enforcing diff_method="adjoint" and explicitly piping the analytical gradient function (gradient_fn = qml.grad(cost_fn)) directly to the SciPy optimizer.

Status: Identified as the optimal architecture. All core feature updates, physics patches, and scaling benchmarks were carried out on this version.

Version 3: JAX-JIT Compilation (Architectural Limit)
-> Objective: Integrate jax and jax.numpy wrapped in @jax.jit decorators to maximize GPU graph execution speed, utilizing array translation bridges (scipy_cost / scipy_grad) to pipe JAX tensors back to SciPy.

-> The Problem: When scaled to large molecular systems (e.g., Tetracene), memory usage skyrocketed to 62 GB and execution wall-clock time bloated to over 7 hours.

-> Root Cause Analysis: Discovered a fundamental limit of applying Just-In-Time (JIT) compilation to active self-consistent Hybrid Quantum Embedding loops. Because the classical embedding environment updates and physically modifies the molecular Hamiltonian at every single iteration, JAX was forced to discard the compiled 62 GB execution graph and re-compile the massive 3,500+ gate quantum circuit from scratch every iteration.

Final Production Workflows (Version 2)

Following the validation of Version 2 as the most stable and efficient framework, the implementation was split into two primary operational folders to handle specific ansatz configurations and physics corrections:

1. PennyLane Version 2/UCCSD/
Contains the core production benchmarking code using the standard Unitary Coupled Cluster Singles and Doubles (UCCSD) ansatz.

Spin Hamiltonian Penalty Code Patch: A custom mathematical patch integrated across all execution files to handle active-space spin-state constraints during convergence loops.
Cross-Platform Comparison Suites: Includes identical codebases written for PennyLane GPU (lightning.gpu), PennyLane CPU, and Qiskit CPU. This suite allows for direct validation that:
1. Numerical results and energy outputs remain virtually identical across PennyLane and Qiskit.
2. Scaling metrics accurately reflect the speedups achieved by offloading exact adjoint gradients to the GPU.

2. PennyLane Version 2/qUCCSD/
An alternative workflow evaluating the quadratic UCCSD (qUCCSD) ansatz variant to study ansatz-dependent convergence behaviors.

Maintains the exact same cross-platform evaluation framework across PennyLane GPU, PennyLane CPU, and Qiskit CPU.
Features the identical Active Space Spin Penalty Patch ported into the qUCCSD circuit structure to ensure consistent physics across all tests.

Current Project Status: This work is currently incomplete; the ultimate goal of achieving superior GPU performance acceleration remains unfulfilled, and the cross-framework validation required to definitively confirm that Qiskit and PennyLane yield identical numerical results has not yet been finalized.