#%%
import numpy as np
import glob
from ase.io.xsf import read_xsf, write_xsf

def calculate_electron_density(xsf_files):
    """Calculate electron density from Wannier functions."""
    density = None
    atoms = None
    origin = None
    span_vectors = None

    for xsf_file in xsf_files:
        try:
            with open(xsf_file, 'r') as f:
                # Read XSF file using ASE
                data, file_origin, file_span_vectors, file_atoms = read_xsf(f, read_data=True)
            
            print(f"\nSuccessfully read {xsf_file}")
            print(f"Shape of data: {data.shape}")
            print(f"Origin: {file_origin}")
            print("Cell:")
            print(file_atoms.cell)
            
            if density is None:
                density = np.zeros_like(data)
                origin = file_origin
                span_vectors = file_span_vectors
                atoms = file_atoms
            density += np.abs(data)**2
        except Exception as e:
            print(f"Error reading {xsf_file}:")
            print(f"  {str(e)}")
            continue

    if density is None:
        raise ValueError("Failed to read any valid XSF files.")

    # Multiply by 2 for spin degeneracy
    density *= 2

    return density, origin, span_vectors, atoms

def main():
    # Get all .xsf files in the current directory
    xsf_files = glob.glob('wannier*.xsf')
    
    if not xsf_files:
        print("No .xsf files found in the current directory.")
        return

    print(f"Found {len(xsf_files)} Wannier function files.")

    try:
        # Calculate electron density
        density, origin, span_vectors, atoms = calculate_electron_density(xsf_files)

        # Save data in NumPy format
        np.save('electron_density.npy', density)
        np.save('grid_info.npy', {'origin': origin, 'span_vectors': span_vectors})

        # Write electron density to XSF file using ASE
        with open('electron_density.xsf', 'w') as f:
            write_xsf(f, [atoms], data=density)

        print("\nFinal results:")
        print(f"Electron density shape: {density.shape}")
        print(f"Origin: {origin}")
        print("Cell:")
        print(atoms.cell)
        print(f"Number of atoms: {len(atoms)}")
        print("Atomic numbers:")
        #Print atomic numbers
        print(atoms.get_atomic_numbers())

        print("\nElectron density saved to 'electron_density.xsf' and 'electron_density.npy'")
        print("Grid information saved to 'grid_info.npy'")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()