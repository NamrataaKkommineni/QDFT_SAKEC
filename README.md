### Repository Structure

```mermaid
graph LR
    Root["Computational Chemistry & Quantum Computing Research Repository"]
    
    %% Anion/Cation Studies (Red)
    Root --- A["Anion/Cation Studies"]
    style A stroke:#EF4444,stroke-width:3px
    A --- A1["anion_otherFunctional.py"]
    A --- A2["anion_tuned_spin.py"]
    A --- A3("Purpose: Anion/Cation calculations using different functionals and spin tuning approaches")




    %% DFT+FCI Calculations (Blue)
    Root --- B["DFT+FCI Calculations"]
    style B stroke:#3B82F6,stroke-width:3px
    B --- B1["Energy + Density"]
    B1 --- B1_1["FCI_ChupinDIIS_NoSpinProtection.py"]
    B1 --- B1_2["FCI_spinFilter.py"]
    B1 --- B1_3["FCI_spinHamiltonian.py"]
    B1 --- B1_4["FCI_spinHamiltonian_LDA-RS.py"]
    B1 --- B1_5["FCI_tuned_spinHamiltonian.py"]
    B1 --- B1_6("Purpose: Energy and density calculations using FCI with spin-related corrections and Hamiltonian modifications")
    
    B --- B2["Energy Calculations"]
    B2 --- B2_1["DFT+FCI_LDA-RS.py"]
    B2 --- B2_2["DFT+FCI_LDA.py"]
    B2 --- B2_3["FCI_camb3lyp_tuned.py"]
    B2 --- B2_4("Purpose: Energy calculations combining DFT and FCI methods")
    
    %% Exact Codes (Yellow)
    Root --- C["Exact Codes"]
    style C stroke:#EAB308,stroke-width:3px
    C --- C1["ExactCode_v1.py"]
    C --- C2["ExactCode_v2.py"]
    
    %% Geometry Optimization (Green)
    Root --- D["Geometry Optimization"]
    style D stroke:#10B981,stroke-width:3px
    D --- D1["CodeOptimizedGeom.py"]
    D --- D2["allOptimizedGeom.py"]
    D --- D3("Purpose: Molecular geometry optimization workflows")
    
    %% VQE+DFT(QDFT) (Orange)
    Root --- E["VQE+DFT(QDFT)"]
    style E stroke:#F97316,stroke-width:3px
    E --- E1["A. Energy+Density"]
    E1 --- E1_1["LDA-RS_MP2_spin.py"]
    E1 --- E1_2["OtherFunctional_MP2_spin.py"]
    E1 --- E1_3["tuned_MP2.py"]
    E1 --- E1_4["tuned_MP2_spin.py"]
    E1 --- E1_5("Purpose: Functional benchmarking")
    
    E --- E2["B. Energy"]
    E2 --- E2_1["IITB Ansatz"]
    E2_1 --- E2_1a["IITB_Ansatz.py"]
    E2_1 --- E2_1b["VQE_camB3lyp_tuned.py"]
    E2_1 --- E2_1c("Purpose: Custom VQE ansatz implementation")
    E2 --- E2_2["UCCSD Ansatz"]
    E2_2 --- E2_2a("Purpose: UCCSD-based VQE calculations")
    
    E --- E3["C. MP2Init"]
    E3 --- E3_1["DFT+VQE_MP2_init.py"]
    E3 --- E3_2("Purpose: MP2 initialization for VQE workflows")
    
    E --- E4["D. HomoLumo"]
    E4 --- E4_1["HomoLumo_v1.py"]
    E4 --- E4_2["HomoLumo_v2.py"]
    E4 --- E4_3("Purpose: HOMO-LUMO gap calculations")
    
    E --- E5["E. Profiling"]
    E5 --- E5_1["ProfiledCode_old_Basic.py"]
    E5 --- E5_2["ProfiledCode_new_Advanced.py"]
    E5 --- E5_3("Purpose: Runtime profiling")
    
    E --- E6["F. ROKS_RKS"]
    E6 --- E6_1["LDA-RS_RKS.py"]
    E6 --- E6_2["LDA-RS_ROKS.py"]
    E6 --- E6_3["OtherFunctional_RKS.py"]
    E6 --- E6_4["OtherFunctionals_ROKS.py"]
    E6 --- E6_5("Purpose: RKS and ROKS calculations Functional comparisons")
    
    %% Quantum IBM (Purple)
    Root --- F["Quantum IBM"]
    style F stroke:#A855F7,stroke-width:3px
    F --- F1["Quantum Time Activation"]
    F1 --- F1_1["Classical.yml"]
    F1 --- F1_2["Quantum.yml"]
    F1 --- F1_3["IBM_Reset.py"]
    F1 --- F1_4["Measure.py"]
    F1 --- F1_5["WS_Classical.py"]
    F1 --- F1_6["WS_Quantum.py"]
    F1 --- F1_7("Purpose: Contains codes such as IBM_Reset.py to setup API token to connect to IBM Server. Measure.py to measure circuit depth, etc(Use quantum env) Classical to perform classical iteration on classical env.")
    
    F --- F2["FE Quantum"]
    F2 --- F2_1["FE_Quantum_v1.py"]
    F2 --- F2_2["FE_Quantum_v2.py"]
    F2 --- F2_3("Purpose: Quantum hardware codes to run on IBM Quantum Platform")
```


[Link Text](https://whimsical.com/shreyas288/qdft-sakec-WutykcUTYDzyrQFXDtghrp)
