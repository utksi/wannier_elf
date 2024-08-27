#%%
import numpy as np
import glob
from ase.io.xsf import read_xsf, write_xsf

def calculate_kinetic_energy_density(wavefunctions, grid_spacing):
    """Calculate kinetic energy density from wavefunctions."""
    kinetic_energy_density = np.zeros_like(wavefunctions[0])
    for wf in wavefunctions:
        # Compute the gradient of the wavefunction
        gradient = np.gradient(wf, *grid_spacing)
        # Sum the squares of the gradient components
        kinetic_energy_density += 0.5 * sum(g**2 for g in gradient)
    return kinetic_energy_density

def read_wannier_functions(xsf_files):
    """Read Wannier functions from XSF files."""
    wavefunctions = []
    origin = None
    span_vectors = None
    atoms = None
    for xsf_file in xsf_files:
        try:
            with open(xsf_file, 'r') as f:
                # Read XSF file using ASE
                data, file_origin, file_span_vectors, file_atoms = read_xsf(f, read_data=True)
                wavefunctions.append(data)
                origin = file_origin
                span_vectors = file_span_vectors
                atoms = file_atoms
        except Exception as e:
            print(f"Error reading {xsf_file}: {str(e)}")
            continue
    return wavefunctions, origin, span_vectors, atoms

def main():
    # Get all .xsf files in the current directory
    xsf_files = glob.glob('wannier*.xsf')
    
    if not xsf_files:
        print("No .xsf files found in the current directory.")
        return

    print(f"Found {len(xsf_files)} Wannier function files.")

    try:
        # Read Wannier functions
        wavefunctions, origin, span_vectors, atoms = read_wannier_functions(xsf_files)

        # Calculate grid spacing
        grid_spacing = np.linalg.norm(span_vectors, axis=1) / wavefunctions[0].shape

        # Calculate kinetic energy density
        kinetic_energy_density = calculate_kinetic_energy_density(wavefunctions, grid_spacing)

        # Save kinetic energy density
        np.save('kinetic_energy_density.npy', kinetic_energy_density)

        # Write kinetic energy density to XSF file
        with open('kinetic_energy_density.xsf', 'w') as f:
            write_xsf(f, [atoms], data=kinetic_energy_density, origin=origin, span_vectors=span_vectors)

        print("\nKinetic energy density saved to 'kinetic_energy_density.xsf' and 'kinetic_energy_density.npy'")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()