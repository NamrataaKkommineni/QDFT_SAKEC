## Geometry Optimization Using PySCF

Before running the main workflow, obtain the molecular geometry from a database such as PubChem and perform a geometry optimization (geometry relaxation) using PySCF. The optimized structure can then be used in subsequent quantum chemistry calculations.

### Example: CO₂ Geometry Optimization

```python
from pyscf import gto, dft
from pyscf.geomopt.geometric_solver import optimize

# ------------------------------------------------------------------
# Step 1: Define the initial molecular geometry
# Coordinates below were obtained from PubChem.
# Replace these coordinates with those of your molecule.
# ------------------------------------------------------------------
mol = gto.M(
    atom="""
    O   0.0000   0.0000  -1.1970
    C   0.0000   0.0000   0.0000
    O   0.0000   0.0000   1.1970
    """,
    basis="6-31g*",
    unit="Angstrom",
    symmetry=True
)

# ------------------------------------------------------------------
# Step 2: Set up a DFT calculation
# ------------------------------------------------------------------
mf = dft.RKS(mol)
mf.xc = "B3LYP"

# ------------------------------------------------------------------
# Step 3: Optimize (relax) the molecular geometry
# ------------------------------------------------------------------
optimized_mol = optimize(mf)

# ------------------------------------------------------------------
# Step 4: Print optimized coordinates in Angstrom
# ------------------------------------------------------------------
BOHR_TO_ANGSTROM = 0.529177

coords_angstrom = optimized_mol.atom_coords() * BOHR_TO_ANGSTROM

print("\nOptimized Geometry (Angstrom):\n")

for i in range(optimized_mol.natm):
    symbol = optimized_mol.atom_symbol(i)
    x, y, z = coords_angstrom[i]
    print(f"{symbol:2s}  {x:10.6f}  {y:10.6f}  {z:10.6f}")
```

### Converting the Optimized Geometry for PySCFDriver

If your workflow requires a geometry string for `PySCFDriver`, you can generate it directly from the optimized structure:

```python
geometry_string = "; ".join(
    f"{optimized_mol.atom_symbol(i)} "
    f"{coords_angstrom[i,0]:.6f} "
    f"{coords_angstrom[i,1]:.6f} "
    f"{coords_angstrom[i,2]:.6f}"
    for i in range(optimized_mol.natm)
)

print("\nGeometry for PySCFDriver:\n")
print(f'geometry = "{geometry_string}"')
```

### Notes

1. Obtain the initial molecular geometry from PubChem (or another molecular database).
2. Replace the example CO₂ coordinates with the coordinates of your target molecule.
3. Run the geometry optimization to obtain a relaxed structure.
4. Use the optimized geometry for VQE and other simulations.
5. Functional used is B3LYP.
6. Some molecules optimized geometries are already present in allOptimizedGeom.py file.
