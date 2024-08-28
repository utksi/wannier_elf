#%%
import numpy as np
from ase.io.xsf import read_xsf, write_xsf

def calculate_elf(pauli_ke_density, heg_ke_density):
    """Calculate the Electron Localization Function (ELF)."""
    # Calculate ELF using the given formula
    elf = 1 / (1 + (pauli_ke_density / heg_ke_density)**2)
    return elf

def main():
    # Load the Pauli kinetic energy density from the .npy file
    pauli_ke_density = np.load('pauli_ke_density.npy')

    # Load the HEG kinetic energy density from the .npy file
    heg_ke_density = np.load('heg_ke_density.npy')

    # Open the electron density XSF file and read the metadata
    with open('electron_density.xsf', 'r') as f:
        _, origin, span_vectors, atoms = read_xsf(f, read_data=True)

    # Calculate the ELF
    elf = calculate_elf(pauli_ke_density, heg_ke_density)

    # Save ELF to .npy file
    np.save('elf.npy', elf)

    # Write ELF to XSF file
    with open('elf.xsf', 'w') as f:
        write_xsf(f, [atoms], data=elf, origin=origin, span_vectors=span_vectors)

    print("Electron Localization Function saved to 'elf.xsf' and 'elf.npy'")

if __name__ == "__main__":
    main()