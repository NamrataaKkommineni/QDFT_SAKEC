from pyscf import gto, dft
from pyscf.geomopt.geometric_solver import optimize
import numpy as np

co2 = gto.M(
    atom="""
    O   0.0000   0.0000  -1.1970
    C   0.0000   0.0000   0.0000
    O   0.0000   0.0000   1.1970
    """,
    basis="6-31g*",
    unit="Angstrom",
    symmetry=True
)

mf = dft.RKS(co2)
mf.xc = "B3LYP"

co2_eq = optimize(mf)

# Convert Bohr to Angstrom
bohr_to_ang = 0.529177
coords_bohr = co2_eq.atom_coords()
coords_ang = coords_bohr * bohr_to_ang

symbols = [co2_eq.atom_symbol(i) for i in range(co2_eq.natm)]

# Build geometry string
geometry_string = ""
for atom, coord in zip(symbols, coords_ang):
    geometry_string += f"{atom}  {coord[0]:.6f}  {coord[1]:.6f}  {coord[2]:.6f}; "

# Remove last semicolon space
geometry_string = geometry_string.rstrip("; ")

print("\nGeometry for PySCFDriver:\n")
print("geometry_CO2 = (")
print(f'    "{geometry_string}"')
print(")")