import numpy as np  # noqa: F401
from ase.io.xsf import read_xsf, write_xsf

def calculate_elf(tau_p, tau_h):
    """Calculate the Electron Localization Function (ELF)."""
    return 1 / (1 + (tau_p / tau_h)**2)

def main():
    # Load Pauli kinetic energy density
    tau_p, origin, cell, atoms, atom_types = read_xsf('pauli_kinetic_energy_density.xsf')
    
    # Load homogeneous electron gas kinetic energy density
    tau_h, _, _, _, _ = read_xsf('homogeneous_electron_gas_ke_density.xsf')
    
    print(f"Loaded Pauli kinetic energy density shape: {tau_p.shape}")
    print(f"Loaded homogeneous electron gas kinetic energy density shape: {tau_h.shape}")
    
    # Calculate ELF
    elf = calculate_elf(tau_p, tau_h)
    
    # Write ELF to XSF file
    write_xsf('electron_localization_function.xsf', elf, origin, cell, atoms, atom_types)
    
    print("Electron Localization Function (ELF) saved to 'electron_localization_function.xsf'")

if __name__ == "__main__":
    main()