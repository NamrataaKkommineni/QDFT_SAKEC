# ===============================================================
#  HF, DFT, CCSD, CASCI, CASSCF reference energies + timings
#  Using PySCF via Qiskit Nature
# ===============================================================
import pyscf
#pyscf.__config__.B3LYP_WITH_VWN5 = True
import time
from qiskit_nature.second_q.drivers import PySCFDriver, MethodType
from qiskit_nature.settings import settings
from pyscf import cc, mcscf

# Optional settings
settings.use_symmetry_reduced_integrals = True

# === 1. Define molecule ===
geometry01="H 0.0 0.0 0.0; H 0.0 0.0 0.7414" # hydrogen H2

propane = (
    "C   0.000000   0.000000   0.586866; "
    "C  -0.000000   1.277427  -0.259609; "
    "C   0.000000  -1.277427  -0.259609; "
    "H   0.877797   0.000000   1.247354; "
    "H  -0.877797  -0.000000   1.247354; "
    "H  -0.000000   2.176115   0.368129; "
    "H   0.000000  -2.176115   0.368129; "
    "H   0.884702   1.322572  -0.907060; "
    "H  -0.884702   1.322572  -0.907060; "
    "H  -0.884702  -1.322572  -0.907060; "
    "H   0.884702  -1.322572  -0.907060"
)

pentane = (
    "C   0.000000   0.000000   0.316190; "
    "C  -0.000000   1.284088  -0.523531; "
    "C   0.000000  -1.284088  -0.523531; "
    "C  -0.000000   2.561094   0.323584; "
    "C   0.000000  -2.561094   0.323584; "
    "H   0.878693   0.000000   0.978795; "
    "H  -0.878693  -0.000000   0.978795; "
    "H   0.878229   1.283604  -1.184936; "
    "H  -0.878229   1.283604  -1.184936; "
    "H  -0.878229  -1.283604  -1.184936; "
    "H   0.878229  -1.283604  -1.184936; "
    "H  -0.000000   3.459308  -0.304697; "
    "H   0.000000  -3.459308  -0.304697; "
    "H  -0.884699   2.607266   0.970763; "
    "H   0.884699   2.607266   0.970763; "
    "H   0.884699  -2.607266   0.970763; "
    "H  -0.884699  -2.607266   0.970763"
)

heptane = (
    "C   0.000000   0.000000   0.494887; "
    "C  -0.000000   1.283886  -0.345225; "
    "C   0.000000  -1.283886  -0.345225; "
    "C  -0.000000   2.567996   0.494821; "
    "C   0.000000  -2.567996   0.494821; "
    "C  -0.000000   3.844938  -0.352381; "
    "C   0.000000  -3.844938  -0.352381; "
    "H  -0.878564  -0.000000   1.157356; "
    "H   0.878564   0.000000   1.157356; "
    "H  -0.878545   1.285024  -1.007836; "
    "H   0.878545   1.285024  -1.007836; "
    "H   0.878545  -1.285024  -1.007836; "
    "H  -0.878545  -1.285024  -1.007836; "
    "H   0.878117   2.567474   1.156358; "
    "H  -0.878117   2.567474   1.156358; "
    "H  -0.878117  -2.567474   1.156358; "
    "H   0.878117  -2.567474   1.156358; "
    "H  -0.000000   4.743228   0.275817; "
    "H   0.884682   3.891077  -0.999631; "
    "H  -0.884682   3.891077  -0.999631; "
    "H   0.000000  -4.743228   0.275817; "
    "H  -0.884682  -3.891077  -0.999631; "
    "H   0.884682  -3.891077  -0.999631"
)

nonane = (
    "C   0.000000   0.000000   0.361928; "
    "C  -0.000000   1.283829  -0.478659; "
    "C   0.000000  -1.283829  -0.478659; "
    "C  -0.000000   2.567719   0.361516; "
    "C   0.000000  -2.567719   0.361516; "
    "C  -0.000000   3.851848  -0.478565; "
    "C   0.000000  -3.851848  -0.478565; "
    "C  -0.000000   5.128779   0.368731; "
    "C   0.000000  -5.128779   0.368731; "
    "H   0.878531   0.000000   1.024319; "
    "H  -0.878531  -0.000000   1.024319; "
    "H   0.878565   1.283794  -1.141071; "
    "H  -0.878565   1.283794  -1.141071; "
    "H  -0.878565  -1.283794  -1.141071; "
    "H   0.878565  -1.283794  -1.141071; "
    "H   0.878575   2.568865   1.024085; "
    "H  -0.878575   2.568865   1.024085; " 
    "H  -0.878575  -2.568865   1.024085; "
    "H   0.878575  -2.568865   1.024085; "
    "H  -0.878153   3.851350  -1.140039; "
    "H   0.878153   3.851350  -1.140039; "
    "H   0.878153  -3.851350  -1.140039; "
    "H  -0.878153  -3.851350  -1.140039; "
    "H  -0.000000   6.027110  -0.259415; "
    "H  -0.884687   5.174845   1.015980; "
    "H   0.884687   5.174845   1.015980; "
    "H   0.000000  -6.027110  -0.259415; "
    "H   0.884687  -5.174845   1.015980; "
    "H  -0.884687  -5.174845   1.015980  "
)

current_molecule = pentane 
molecule_name = "Pentane C5H12"
basis_set = "6-31g*"

print("======================================================")
print(f"Molecule: {molecule_name} | Basis: {basis_set}")
print("======================================================")

# === 2. HF Calculation ===
start_hf = time.time()
driver_hf = PySCFDriver(atom=current_molecule, basis=basis_set, method=MethodType.RHF)
problem_hf = driver_hf.run()
mf = driver_hf._calc
mol = driver_hf._mol
end_hf = time.time()

E_hf = mf.e_tot
print(f"RHF Energy: {E_hf:.8f} Ha")
print(f"Time (HF): {end_hf - start_hf:.3f} s\n")

# === 3. MP2 Calculation ===
from pyscf import mp

start_mp2 = time.time()

mp2_calc = mp.MP2(mf)      # RHF → MP2
E_corr_mp2, t2 = mp2_calc.kernel()

end_mp2 = time.time()

E_mp2 = E_hf + E_corr_mp2

print(f"MP2 Correlation Energy: {E_corr_mp2:.8f} Ha")
print(f"MP2 Total Energy:       {E_mp2:.8f} Ha")
print(f"Time (MP2): {end_mp2 - start_mp2:.3f} s\n")
# === 3. DFT Calculation ===
# You can change 'lda,vwn' → 'b3lyp', 'pbe', etc.
start_dft = time.time()
driver_dft = PySCFDriver(atom=current_molecule, basis=basis_set, method=MethodType.RKS, xc_functional="lda,vwn")
problem_dft = driver_dft.run()
mf_dft = driver_dft._calc
end_dft = time.time()

E_dft = mf_dft.e_tot
print(f"DFT (LDA,VWN) Energy: {E_dft:.8f} Ha")
print(f"Time (DFT): {end_dft - start_dft:.3f} s\n")

# === 4. CCSD ===
start_ccsd = time.time()
cc_calc = cc.CCSD(mf)
cc_energy, t1, t2 = cc_calc.kernel()
E_ccsd_total = mf.e_tot + cc_energy
end_ccsd = time.time()

print(f"CCSD Total Energy: {E_ccsd_total:.8f} Ha")
print(f"Time (CCSD): {end_ccsd - start_ccsd:.3f} s\n")

# === 5. Define Active Space ===
n_active_orb = 4
n_active_elec = 2
print("------------------------------------------------------")
print(f"Active Space: {n_active_elec} electrons in {n_active_orb} orbitals")
print("------------------------------------------------------")

# === 6. CASCI ===
start_casci = time.time()
mc_casci = mcscf.CASCI(mf, n_active_orb, n_active_elec)
E_casci = mc_casci.kernel()[0]
end_casci = time.time()
print(f"CASCI({n_active_elec}e, {n_active_orb}o) Energy: {E_casci:.8f} Ha")
print(f"Time (CASCI): {end_casci - start_casci:.3f} s\n")

# === 7. CASSCF ===
start_casscf = time.time()
mc_casscf = mcscf.CASSCF(mf, n_active_orb, n_active_elec)
E_casscf = mc_casscf.kernel()[0]
end_casscf = time.time()
print(f"CASSCF({n_active_elec}e, {n_active_orb}o) Energy: {E_casscf:.8f} Ha")
print(f"Time (CASSCF): {end_casscf - start_casscf:.3f} s\n")

print("======================================================")
print("All reference methods computed successfully.")
print("======================================================")
