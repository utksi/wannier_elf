#%%
import numpy as np
from scipy.ndimage import gaussian_filter
from ase.io.xsf import read_xsf, write_xsf

def calculate_elf(pauli_ke_density, heg_ke_density, smooth=True, sigma=0.5, epsilon=1e-10):
    """
    Calculate the Electron Localization Function (ELF) with improvements.
    
    Parameters:
    pauli_ke_density (np.array): Pauli kinetic energy density
    heg_ke_density (np.array): Homogeneous electron gas kinetic energy density
    smooth (bool): Apply Gaussian smoothing to the result
    sigma (float): Sigma for Gaussian smoothing
    epsilon (float): Small value for regularization
    
    Returns:
    np.array: Calculated ELF
    """
    # Ensure high precision
    pauli_ke_density = pauli_ke_density.astype(np.float64)
    heg_ke_density = heg_ke_density.astype(np.float64)
    
    # Calculate ELF with regularization
    elf = 1.0 / (1.0 + ((pauli_ke_density + epsilon) / (heg_ke_density + epsilon))**2)
    
    # Apply smoothing if requested
    if smooth:
        elf = gaussian_filter(elf, sigma=sigma)
    
    return elf

def main():
    # Load the Pauli kinetic energy density
    pauli_ke_density = np.load('pauli_ke_density.npy')
    
    # Load the HEG kinetic energy density
    heg_ke_density = np.load('heg_ke_density.npy')
    
    # Load metadata from a previous XSF file (e.g., electron density)
    with open('electron_density.xsf', 'r') as f:
        _, origin, span_vectors, atoms = read_xsf(f, read_data=True)
    
    # Calculate ELF
    elf = calculate_elf(pauli_ke_density, heg_ke_density, smooth=True, sigma=0.5, epsilon=1e-10)
    
    # Save ELF to .npy file
    np.save('elf.npy', elf)
    
    # Write ELF to XSF file
    with open('elf.xsf', 'w') as f:
        write_xsf(f, [atoms], data=elf, origin=origin, span_vectors=span_vectors)
    
    print("ELF calculation completed.")
    print("ELF saved to 'elf.npy' and 'elf.xsf'")
    
    # Additional checks and information
    print(f"ELF shape: {elf.shape}")
    print(f"ELF min value: {elf.min()}")
    print(f"ELF max value: {elf.max()}")
    print(f"ELF mean value: {elf.mean()}")
    print(f"ELF standard deviation: {elf.std()}")
    
    # Check for any NaN or inf values
    if np.any(np.isnan(elf)) or np.any(np.isinf(elf)):
        print("Warning: ELF contains NaN or inf values!")
    
    # Optionally, you can save both smoothed and unsmoothed versions
    elf_unsmoothed = calculate_elf(pauli_ke_density, heg_ke_density, smooth=False)
    np.save('elf_unsmoothed.npy', elf_unsmoothed)
    print("Unsmoothed ELF saved to 'elf_unsmoothed.npy'")

if __name__ == "__main__":
    main()