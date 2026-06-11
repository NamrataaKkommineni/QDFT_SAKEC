# ===============================================================
# HF, MP2, DFT (various functionals), CCSD, CCSD(T) reference energies + timings
# Pure PySCF - Using YOUR range separation method with PySCFDriver
# Supports open-shell systems (ROHF/ROKS) 
# Change geometry ONLY at the top
# ===============================================================
import time
from qiskit_nature.second_q.drivers import PySCFDriver, MethodType
from pyscf import cc, mp, mcscf

# === USER INPUT: Change ONLY this geometry ===
geometry = """
C  5.890096  -0.722620  0.000000
C  5.890096  0.722620  0.000000
C  3.442033  -0.722608  0.000000
C  3.442033  0.722608  0.000000
C  4.666079  -1.403534  0.000000
C  4.666079  1.403534  0.000000
C  7.146040  -1.407540  0.000000
C  7.146040  1.407540  0.000000
C  2.186103  -1.407521  0.000000
C  2.186103  1.407521  0.000000
C  8.327468  -0.713406  0.000000
C  8.327468  0.713406  0.000000
C  1.004660  -0.713397  0.000000
C  1.004660  0.713397  0.000000
H  4.666140  -2.491880  0.000000
H  4.666140  2.491880  0.000000
H  7.143168  -2.495025  0.000000
H  7.143168  2.495025  0.000000
H  2.188982  -2.495007  0.000000
H  2.188982  2.495007  -0.000000
H  9.274249  -1.246632  0.000000
H  9.274249  1.246632  0.000000
H  0.057891  -1.246645  0.000000
H  0.057891  1.246645  -0.000000
"""
basis_set = "6-31g*" # or "sto-3g"  # or "6-31g*"
charge = 0
spin = 0  # 0=closed shell singlet, 1=doublet, 2=triplet, etc.
omega = 5.0  # Range separation parameter
n_active_orb = 6
n_active_elec = 6

print("=== Benchmark: PySCF Electronic Structure Methods ===\n")
print(f"Geometry:\n{geometry}\n")
print(f"Basis: {basis_set}, Charge: {charge}, Spin: {spin}, Omega: {omega}\n")

# === 1. HF Calculation ===
print("=== 1. HF Calculation ===")
start_hf = time.time()
if spin == 0:
    driver_hf = PySCFDriver(atom=geometry.strip(), basis=basis_set, method=MethodType.RHF, charge=charge, spin=spin)
else:
    driver_hf = PySCFDriver(atom=geometry.strip(), basis=basis_set, method=MethodType.ROHF, charge=charge, spin=spin)
problem_hf = driver_hf.run()
mf_hf = driver_hf._calc
end_hf = time.time()

E_hf = mf_hf.e_tot
print(f"RHF/ROHF Energy: {E_hf:.8f} Ha")
print(f"Time (HF): {end_hf - start_hf:.3f} s\n")

# === 2. MP2 Calculation ===
print("=== 2. MP2 Calculation ===")
start_mp2 = time.time()
mp2_calc = mp.MP2(mf_hf)
E_corr_mp2, t2 = mp2_calc.kernel()
end_mp2 = time.time()

E_mp2 = E_hf + E_corr_mp2
print(f"MP2 Correlation Energy: {E_corr_mp2:.8f} Ha")
print(f"MP2 Total Energy:       {E_mp2:.8f} Ha")
print(f"Time (MP2): {end_mp2 - start_mp2:.3f} s\n")

# === 3. DFT Methods - Using YOUR PySCFDriver style ===
dft_methods = {
    "LDA": "lda,vwn",
    "PBE": "pbe,pbe",
    "B3LYP": "b3lyp", 
    "wB97X": "wb97x"
}

dft_energies = {}
dft_times = {}

for name, xc in dft_methods.items():
    print(f"=== 3.{list(dft_methods.keys()).index(name)+1} DFT({name}) ===")
    start_dft = time.time()
    method_type = MethodType.RKS if spin == 0 else MethodType.ROKS
    driver_dft = PySCFDriver(
        atom=geometry.strip(),
        basis=basis_set, 
        method=method_type,
        charge=charge,
        spin=spin,
        xc_functional=xc
    )
    problem_dft = driver_dft.run()
    mf_dft = driver_dft._calc
    end_dft = time.time()
    
    dft_energies[name] = mf_dft.e_tot
    dft_times[name] = end_dft - start_dft
    print(f"DFT({name}) Energy: {mf_dft.e_tot:.8f} Ha")
    print(f"Time (DFT-{name}): {dft_times[name]:.3f} s\n")

# === 3.5 DFT(LDA-RS) - YOUR range separation method ===
print("=== 3.5 DFT(LDA-RS) - Range Separated ===")
start_dft_rs = time.time()
method_type_rs = MethodType.RKS if spin == 0 else MethodType.ROKS
driver_dft_rs = PySCFDriver(
    atom=geometry.strip(),
    basis=basis_set,
    method=method_type_rs,
    charge=charge,
    spin=spin,
    xc_functional=f"ldaerf + lr_hf({omega})",
    xcf_library="xcfun"  # Required for range-separated functionals
)
problem_dft_rs = driver_dft_rs.run()
mf_dft_rs = driver_dft_rs._calc
end_dft_rs = time.time()

dft_energies["LDA-RS"] = mf_dft_rs.e_tot
dft_times["LDA-RS"] = end_dft_rs - start_dft_rs
print(f"DFT(LDA-RS, ω={omega}) Energy: {mf_dft_rs.e_tot:.8f} Ha")
print(f"Time (DFT-LDA-RS): {dft_times['LDA-RS']:.3f} s\n")

# === 4. CCSD ===
print("=== 4. CCSD Calculation ===")
start_ccsd = time.time()
ccsd_calc = cc.CCSD(mf_hf)
# FIXED: Added t1 to catch all three returned values
E_corr_ccsd, t1, t2 = ccsd_calc.kernel() 
end_ccsd = time.time()
E_ccsd = E_hf + E_corr_ccsd
print(f"CCSD Correlation Energy: {E_corr_ccsd:.8f} Ha")
print(f"CCSD Total Energy:       {E_ccsd:.8f} Ha")
print(f"Time (CCSD): {end_ccsd - start_ccsd:.3f} s\n")

# === 5. CCSD(T) ===
print("=== 5. CCSD(T) Calculation ===")
start_ccsdt = time.time()
# FIXED: Calculate perturbative triples directly from the converged CCSD object
E_corr_t = ccsd_calc.ccsd_t() 
end_ccsdt = time.time()
E_ccsdt = E_ccsd + E_corr_t
print(f"CCSD(T) T-correction:    {E_corr_t:.8f} Ha")
print(f"CCSD(T) Total Energy:    {E_ccsdt:.8f} Ha")
print(f"Time (CCSD(T)): {end_ccsdt - start_ccsdt:.3f} s\n")

# === 6. Define Active Space ===
print(f"=== 6. Active Space: {n_active_elec}e in {n_active_orb}o ===")

# === 7. CASCI Calculation ===
print("=== 7. CASCI Calculation ===")
start_casci = time.time()
# Note: Changed 'mf' to 'mf_hf' to match your script
mc_casci = mcscf.CASCI(mf_hf, n_active_orb, n_active_elec)
E_casci = mc_casci.kernel()[0]
end_casci = time.time()
print(f"CASCI({n_active_elec}e, {n_active_orb}o) Energy: {E_casci:.8f} Ha")
print(f"Time (CASCI): {end_casci - start_casci:.3f} s\n")

# === 8. CASSCF Calculation ===
print("=== 8. CASSCF Calculation ===")
start_casscf = time.time()
# Note: Changed 'mf' to 'mf_hf' to match your script
mc_casscf = mcscf.CASSCF(mf_hf, n_active_orb, n_active_elec)
E_casscf = mc_casscf.kernel()[0]
end_casscf = time.time()
print(f"CASSCF({n_active_elec}e, {n_active_orb}o) Energy: {E_casscf:.8f} Ha")
print(f"Time (CASSCF): {end_casscf - start_casscf:.3f} s\n")

# === SUMMARY TABLE ===
print("=== SUMMARY ===")
print(f"{'Method':<12} {'Energy (Ha)':<16} {'Time (s)':<10}")
print("-" * 40)
print(f"{'HF':<12} {E_hf:<16.8f} {(end_hf-start_hf):<10.3f}")
print(f"{'MP2':<12} {E_mp2:<16.8f} {(end_mp2-start_mp2):<10.3f}")
print(f"{'CCSD':<12} {E_ccsd:<16.8f} {(end_ccsd-start_ccsd):<10.3f}")
print(f"{'CCSD(T)':<12} {E_ccsdt:<16.8f} {(end_ccsdt-start_ccsdt):<10.3f}")
print(f"{'CASCI':<12} {E_casci:<16.8f} {(end_casci-start_casci):<10.3f}")
print(f"{'CASSCF':<12} {E_casscf:<16.8f} {(end_casscf-start_casscf):<10.3f}")

print("\nDFT Results:")
for name in dft_methods:
    print(f"  DFT({name}): {dft_energies[name]:.8f} Ha ({dft_times[name]:.3f} s)")
print(f"  DFT(LDA-RS): {dft_energies['LDA-RS']:.8f} Ha ({dft_times['LDA-RS']:.3f} s)")

print("\n✓ All calculations completed using your PySCFDriver style!")
