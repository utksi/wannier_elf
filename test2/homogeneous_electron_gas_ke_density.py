import numpy as np
from ase.io.xsf import read_xsf, write_xsf

def calculate_homogeneous_electron_gas_ke_density(rho):
    """Calculate homogeneous electron gas kinetic energy density."""
    return (3/10) * (3 * np.pi**2)**(2/3) * rho**(5/3)

def main():
    # Load electron density
    rho, origin, cell, atoms, atom_types = read_xsf('electron_density.xsf')
    
    print(f"Loaded electron density shape: {rho.shape}")
    
    # Calculate homogeneous electron gas kinetic energy density
    tau_h = calculate_homogeneous_electron_gas_ke_density(rho)
    
    # Write homogeneous electron gas kinetic energy density to XSF file
    write_xsf('homogeneous_electron_gas_ke_density.xsf', tau_h, origin, cell, atoms, atom_types)
    
    print("Homogeneous electron gas kinetic energy density saved to 'homogeneous_electron_gas_ke_density.xsf'")

if __name__ == "__main__":
    main()