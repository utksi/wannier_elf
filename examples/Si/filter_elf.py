import glob
import numpy as np
from ase.io import write
from ase.io.xsf import iread_xsf
from scipy import ndimage

def pbc_gradient(data, dx, axis):
    grad = np.zeros_like(data)
    slice1 = [slice(None)] * data.ndim
    slice2 = [slice(None)] * data.ndim
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    grad[tuple(slice1)] = (data[tuple(slice1)] - data[tuple(slice2)]) / dx
    grad[tuple(slice2)] = (np.roll(data, -1, axis=axis)[tuple(slice2)] - data[tuple(slice2)]) / dx
    return grad

def pbc_laplacian(data, dx, dy, dz):
    laplacian = (pbc_gradient(pbc_gradient(data, dx, 0), dx, 0) +
                 pbc_gradient(pbc_gradient(data, dy, 1), dy, 1) +
                 pbc_gradient(pbc_gradient(data, dz, 2), dz, 2))
    return laplacian

def calculate_ke_density(wannier_data, span_vectors, electron_density):
    A = 1  # ℏ^2/(2m) but in A.U.
    
    # Calculate grid spacings from span vectors
    dx = np.linalg.norm(span_vectors[0]) / (wannier_data.shape[1] - 1)
    dy = np.linalg.norm(span_vectors[1]) / (wannier_data.shape[2] - 1)
    dz = np.linalg.norm(span_vectors[2]) / (wannier_data.shape[3] - 1)
    
    # First term: −2A∑ψ^{*}∇^2 ψ
    laplacian_term = np.zeros_like(electron_density)
    for wf in wannier_data:
        laplacian_wf = pbc_laplacian(wf, dx, dy, dz)
        laplacian_term += np.real(np.conj(wf) * laplacian_wf)
    T = -2 * A * laplacian_term
    
    # Second term: (A/2) ∇^2 n
    laplacian_density = pbc_laplacian(electron_density, dx, dy, dz)
    T_corr = 0.5 * A * laplacian_density
    
    # Third term: [(A/(4n)] (∇n)^2
    grad_x = pbc_gradient(electron_density, dx, 0)
    grad_y = pbc_gradient(electron_density, dy, 1)
    grad_z = pbc_gradient(electron_density, dz, 2)
    grad_density_squared = grad_x**2 + grad_y**2 + grad_z**2
    T_bos = 0.25 * A * grad_density_squared / np.maximum(electron_density, 1e-12)
    
    # Combine terms as per CASTEP/VASP formula: D = T + T_corr - T_bos
    D = T + T_corr - T_bos
    
    return D

def calculate_elf(electron_density, ke_density, normalize=False, threshold=1e-6):
    # Thomas-Fermi kinetic energy density
    tf_ke_density = 3/10 * (3 * np.pi**2)**(2/3) * electron_density**(5/3)
    
    # ELF formula
    elf = 1 / (1 + (ke_density / np.maximum(tf_ke_density, 1e-12))**2)
    
    # Apply threshold to filter out very small values
    elf[elf < threshold] = 0
    
    if normalize:
        elf = (elf - elf.min()) / (elf.max() - elf.min())
    
    return elf

def analyze_elf(elf, threshold=0.9):
    """Analyze high-value regions of the ELF"""
    high_elf = elf > threshold
    labeled, num_features = ndimage.label(high_elf)
    
    print(f"Number of high-ELF regions (>{threshold}): {num_features}")
    
    for i in range(1, num_features + 1):
        region = labeled == i
        volume = region.sum()
        print(f"Region {i}: Volume = {volume} grid points")

# Main script
if __name__ == "__main__":
    # Get all Wannier function XSF files
    wannier_files = glob.glob('./wannier*.xsf')
    print(f"Found {len(wannier_files)} XSF files")

    # Read the data from each file
    wannier_data = []
    atoms = None
    origin = None
    span_vectors = None

    for file in wannier_files:
        print(f"Reading file: {file}")
        with open(file, 'r') as fileobj:
            try:
                images = list(iread_xsf(fileobj, read_data=True))
                data, origin, span_vectors = images[-1]
                atoms = images[0]
                wannier_data.append(data)
                print(f"  Data shape: {data.shape}")
            except Exception as e:
                print(f"  Error reading file {file}: {e}")

    # Convert to numpy array
    wannier_data = np.array(wannier_data)

    print("Shape of wannier_data:", wannier_data.shape)
    print("Shape of span_vectors:", np.array(span_vectors).shape)

    if len(wannier_data) == 0:
        print("No data was read from the XSF files. Please check the file path and content.")
        exit(1)

    # Calculate electron density
    electron_density = np.sum(np.abs(wannier_data)**2, axis=0)
    print("Shape of electron_density:", electron_density.shape)

    try:
        # Calculate kinetic energy density
        ke_density = calculate_ke_density(wannier_data, span_vectors, electron_density)
        
        # Calculate ELF
        elf = calculate_elf(electron_density, ke_density, normalize=False, threshold=1e-6)

        # Analyze high-ELF regions
        analyze_elf(elf, threshold=0.9)

        # Write the ELF to a new XSF file
        output_xsf = 'filtered_elf.xsf'
        write(output_xsf, atoms, format='xsf', data=elf, origin=origin, span_vectors=span_vectors)

        # Save high-ELF regions for visualization
        high_elf = (elf > 0.9).astype(float)
        write('high_elf.xsf', atoms, format='xsf', data=high_elf, origin=origin, span_vectors=span_vectors)

        print(f"ELF calculation completed and written to {output_xsf}")
        print("High-ELF regions written to high_elf.xsf")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Span vectors:", span_vectors)
        print("Please check that the dimensions of your data match the span vectors.")
