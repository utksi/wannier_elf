#%%
import numpy as np
from ase.io.xsf import read_xsf
from ase import Atoms  # noqa: F401
import spglib

def read_wannier_xsf(xsf_file):
    with open(xsf_file, 'r') as f:
        data, origin, span_vectors, atoms = read_xsf(f, read_data=True)
    return data, origin, span_vectors, atoms

def get_symmetry_operations(atoms):
    # Use spglib to get symmetry operations
    lattice = atoms.get_cell()
    positions = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()
    symmetry_dataset = spglib.get_symmetry_dataset((lattice, positions, numbers))
    rotations = symmetry_dataset['rotations']
    translations = symmetry_dataset['translations']
    return rotations, translations

def apply_symmetry_to_scalar_field(data, rotations, translations, grid_shape):
    adjusted_data = np.zeros(grid_shape)
    for rot, trans in zip(rotations, translations):
        # Apply rotation and translation to the data
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                for k in range(grid_shape[2]):
                    # Calculate new position
                    fractional_pos = np.array([i, j, k]) / grid_shape
                    new_pos = np.dot(rot, fractional_pos) + trans
                    new_pos = np.mod(new_pos, 1)  # Ensure periodicity
                    new_pos_index = (new_pos * grid_shape).astype(int)
                    adjusted_data[tuple(new_pos_index)] += data[i, j, k]
    return adjusted_data

def write_adjusted_xsf(filename, grid_data, atoms, origin, span_vectors):
    with open(filename, 'w') as f:
        f.write(" CRYSTAL\n")
        f.write(" PRIMVEC\n")
        for vec in atoms.get_cell():
            f.write(f"  {vec[0]:12.6f} {vec[1]:12.6f} {vec[2]:12.6f}\n")
        f.write(" PRIMCOORD\n")
        f.write(f"  {len(atoms)} 1\n")
        for atom in atoms:
            f.write(f"  {atom.symbol:2s} {atom.position[0]:12.6f} {atom.position[1]:12.6f} {atom.position[2]:12.6f}\n")
        f.write(" BEGIN_BLOCK_DATAGRID_3D\n")
        f.write(" adjusted_wannier_function\n")
        f.write(" BEGIN_DATAGRID_3D_UNKNOWN\n")
        f.write(f"  {grid_data.shape[0]} {grid_data.shape[1]} {grid_data.shape[2]}\n")
        f.write(f"  {origin[0]:12.6f} {origin[1]:12.6f} {origin[2]:12.6f}\n")
        for vec in span_vectors:
            f.write(f"  {vec[0]:12.6f} {vec[1]:12.6f} {vec[2]:12.6f}\n")
        for k in range(grid_data.shape[2]):
            for j in range(grid_data.shape[1]):
                for i in range(grid_data.shape[0]):
                    f.write(f" {grid_data[i,j,k]:12.6E}")
                    if (i+1) % 6 == 0:
                        f.write("\n")
                if grid_data.shape[0] % 6 != 0:
                    f.write("\n")
        f.write(" END_DATAGRID_3D\n")
        f.write(" END_BLOCK_DATAGRID_3D\n")

# Main process
xsf_file = 'wannier90_00001.xsf'

# Read original XSF file
data, origin, span_vectors, atoms = read_wannier_xsf(xsf_file)

# Get symmetry operations using spglib
rotations, translations = get_symmetry_operations(atoms)

# Apply symmetry operations to scalar field
adjusted_data = apply_symmetry_to_scalar_field(data, rotations, translations, data.shape)

# Write the adjusted data to a new XSF file
write_adjusted_xsf('adjusted_periodic_wannier.xsf', adjusted_data, atoms, origin, span_vectors)