# Quantum DFT Embedding Profiler (v3)

A highly instrumented classical-quantum embedding simulation using **PySCF** and **Qiskit Nature**. This script performs density functional theory (DFT) embedding coupled with a Variational Quantum Eigensolver (VQE), specifically optimized to benchmark and profile the exact time-costs of hybrid quantum-classical algorithms.

## 🚀 Overview
While standard Qiskit runtime scripts treat the optimization loop as a black box, `profilev3.py` cracks it open. It provides millisecond-accurate, per-iteration breakdowns of where your compute time is actually going: circuit evaluation, gradient calculation, classical optimizer logic, or Qiskit overhead.

## 🧬 Simulation Specifics: The UCCSD Ansatz
**Note:** This specific script utilizes the **UCCSD (Unitary Coupled Cluster Singles and Doubles)** ansatz, initialized with a Hartree-Fock state. 

*(If you are looking for the version of this profiler that implements the custom **IITB ansatz**, please refer to the alternate script provided in this repository.)*

## ✨ What's New in v3? (Upgrading from v2)

Version 2 successfully timed the macro-steps of the embedding loop (DFT Reference, Integrals, Embedding overhead, and monolithic VQE time). **Version 3 introduces micro-profiling**, splitting the VQE process into its fundamental components.

### 1. Granular VQE Split
In v2, quantum time was a single lump sum (`Quantum_VQE`). In v3, every single VQE iteration is dissected into:
* **Energy Evaluation Time:** Time spent strictly evaluating the ansatz energy.
* **Gradient Evaluation Time:** Time spent calculating gradients (e.g., via finite differences or parameter shifts).
* **Optimizer Overhead:** The classical time the optimizer (L-BFGS-B) spends crunching the numbers to guess the next parameter step.
* **Callback Time:** The time spent inside the classical tracking logic.

### 2. The `TimingEstimator`
To achieve this granularity, v3 introduces a custom drop-in replacement for Qiskit's native `Estimator`. The `TimingEstimator` intercepts every single `.run()` invocation and timestamps it. The callback then reads this log to figure out exactly how many milliseconds were spent communicating with the estimator vs. running classical optimizer math.

### 3. VQE Initialization Tracking
v3 accurately isolates the **VQE Initialization Time** (mapping, tapering, and ansatz preparation) from the actual quantum optimization loop by dynamically stopping the initialization timer the moment the first `eval_count` hits the callback.

### 4. Rich CLI Reporting
v3 prints beautifully formatted, iteration-by-iteration tables directly to your console, allowing you to watch the convergence and timing cost in real-time.

---

## 🛠️ How it Works Under the Hood

The magic of v3 relies on a synchronized dance between the `TimingEstimator` and the `vqe_callback`:

1.  **The Log:** Every time the optimizer requests an energy or gradient, `TimingEstimator.run()` stamps the start/stop time and pushes it to an internal log.
2.  **The Split:** When the `vqe_callback` triggers at the end of an iteration, it looks at all estimator calls made since the *last* callback.
3.  **The Deduction:** Because optimizers like L-BFGS-B typically evaluate the current energy once and then evaluate the surrounding gradients, the script assumes the first call in the log is the `Energy Eval` and the subsequent calls are `Gradient Evals`. 
4.  **The Math:** Total time since the last callback, minus the estimator calls, perfectly isolates the pure `Optimizer` time.

---

## 📊 Expected Output Example

When running the script, you will now see a detailed VQE iteration breakdown table inside every embedding loop:

```text
==========================================================================================
VQE PER-ITERATION BREAKDOWN
==========================================================================================
Iter |    Energy (Ha) |  EnergyEval |   GradEval |  Optimizer |  Callback |  IterTotal
------------------------------------------------------------------------------------------
   1 |    -1.12345678 |       4.2ms |     28.5ms |      1.1ms |     0.0ms |     33.8ms
   2 |    -1.13456789 |       4.1ms |     28.2ms |      0.9ms |     0.0ms |     33.2ms
   3 |    -1.14567890 |       4.0ms |     28.3ms |      1.0ms |     0.0ms |     33.3ms
------------------------------------------------------------------------------------------
 SUM |                |      12.3ms |     85.0ms |      3.0ms |     0.0ms |    100.3ms
==========================================================================================
