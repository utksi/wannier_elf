import numpy as np
from ase.io.xsf import read_xsf, write_xsf

def calculate_kinetic_energy_density(density, cell):
    """Calculate kinetic energy density from electron density."""
    nx, ny, nz = density.shape
    dx, dy, dz = 1/nx, 1/ny, 1/nz
    
    # Calculate gradient in fractional coordinates
    grad_x, grad_y, grad_z = np.gradient(density, dx, dy, dz)
    
    # Convert gradient to Cartesian coordinates
    grad_cart = np.einsum('ij,jklm->iklm', cell, np.array([grad_x, grad_y, grad_z]))
    
    # Calculate kinetic energy density
    ke_density = 0.5 * np.sum(np.abs(grad_cart)**2, axis=0)
    
    return ke_density

def main():
    # Load electron density and grid information
    density, origin, cell, atoms, atom_types = read_xsf('electron_density.xsf')
    
    print(f"Loaded electron density shape: {density.shape}")
    
    # Calculate kinetic energy density
    ke_density = calculate_kinetic_energy_density(density, cell)
    
    # Write kinetic energy density to XSF file
    write_xsf('kinetic_energy_density.xsf', ke_density, origin, cell, atoms, atom_types)
    
    print("Kinetic energy density saved to 'kinetic_energy_density.xsf'")

if __name__ == "__main__":
    main()