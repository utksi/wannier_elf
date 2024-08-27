#%%
import numpy as np
from ase.io.xsf import read_xsf, write_xsf

def calculate_pauli_kinetic_energy_density(electron_density, kinetic_energy_density, grid_spacing):
    """Calculate Pauli kinetic energy density."""
    # Calculate the gradient of the electron density
    grad_rho = np.gradient(electron_density, *grid_spacing)
    grad_rho_squared = sum(g**2 for g in grad_rho)
    
    # Calculate the Pauli kinetic energy density
    pauli_ke_density = kinetic_energy_density - (1/8) * (grad_rho_squared / electron_density)
    
    # Handle division by zero by setting those values to zero
    pauli_ke_density[np.isnan(pauli_ke_density)] = 0
    pauli_ke_density[np.isinf(pauli_ke_density)] = 0
    
    return pauli_ke_density

def main():
    # Open the electron density XSF file and read the data
    with open('electron_density.xsf', 'r') as f:
        data, origin, span_vectors, atoms = read_xsf(f, read_data=True)
        electron_density = data

    # Load the kinetic energy density from the .npy file
    kinetic_energy_density = np.load('kinetic_energy_density.npy')

    # Calculate grid spacing based on span vectors and electron density shape
    grid_spacing = np.linalg.norm(span_vectors, axis=1) / electron_density.shape

    # Calculate Pauli kinetic energy density
    pauli_ke_density = calculate_pauli_kinetic_energy_density(electron_density, kinetic_energy_density, grid_spacing)

    # Save Pauli kinetic energy density to .npy file
    np.save('pauli_ke_density.npy', pauli_ke_density)

    # Write Pauli kinetic energy density to XSF file
    with open('pauli_ke_density.xsf', 'w') as f:
        write_xsf(f, [atoms], data=pauli_ke_density, origin=origin, span_vectors=span_vectors)

    print("Pauli kinetic energy density saved to 'pauli_ke_density.xsf' and 'pauli_ke_density.npy'")

if __name__ == "__main__":
    main()