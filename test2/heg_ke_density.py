#%%
import numpy as np
from ase.io.xsf import read_xsf, write_xsf

def calculate_heg_kinetic_energy_density(electron_density):
    """Calculate the kinetic energy density for a homogeneous electron gas."""
    # HEG kinetic energy density formula
    heg_ke_density = (3/10) * (3 * np.pi**2)**(2/3) * electron_density**(5/3)
    return heg_ke_density

def main():
    # Open the electron density XSF file and read the data
    with open('electron_density.xsf', 'r') as f:
        data, origin, span_vectors, atoms = read_xsf(f, read_data=True)
        electron_density = data

    # Calculate HEG kinetic energy density
    heg_ke_density = calculate_heg_kinetic_energy_density(electron_density)

    # Save HEG kinetic energy density to .npy file
    np.save('heg_ke_density.npy', heg_ke_density)

    # Write HEG kinetic energy density to XSF file
    with open('heg_ke_density.xsf', 'w') as f:
        write_xsf(f, [atoms], data=heg_ke_density, origin=origin, span_vectors=span_vectors)

    print("Homogeneous electron gas kinetic energy density saved to 'heg_ke_density.xsf' and 'heg_ke_density.npy'")

if __name__ == "__main__":
    main()