import numpy as np
from ase.io.xsf import read_xsf, write_xsf

def calculate_pauli_kinetic_energy_density(tau, rho, cell):
    """Calculate Pauli kinetic energy density."""
    nx, ny, nz = rho.shape
    dx, dy, dz = 1/nx, 1/ny, 1/nz
    
    # Calculate gradient in fractional coordinates
    grad_x, grad_y, grad_z = np.gradient(rho, dx, dy, dz)
    
    # Convert gradient to Cartesian coordinates
    grad_cart = np.einsum('ij,jklm->iklm', cell, np.array([grad_x, grad_y, grad_z]))
    
    grad_rho_squared = np.sum(np.abs(grad_cart)**2, axis=0)
    return tau - grad_rho_squared / (8 * rho)

def main():
    # Load kinetic energy density
    tau, origin, cell, atoms, atom_types = read_xsf('kinetic_energy_density.xsf')
    
    # Load electron density
    rho, _, _, _, _ = read_xsf('electron_density.xsf')
    
    print(f"Loaded kinetic energy density shape: {tau.shape}")
    print(f"Loaded electron density shape: {rho.shape}")
    
    # Calculate Pauli kinetic energy density
    tau_p = calculate_pauli_kinetic_energy_density(tau, rho, cell)
    
    # Write Pauli kinetic energy density to XSF file
    write_xsf('pauli_kinetic_energy_density.xsf', tau_p, origin, cell, atoms, atom_types)
    
    print("Pauli kinetic energy density saved to 'pauli_kinetic_energy_density.xsf'")

if __name__ == "__main__":
    main()