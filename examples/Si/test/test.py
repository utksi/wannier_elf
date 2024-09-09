import glob
import numpy as np
from ase.io import write
from ase.io.xsf import iread_xsf

def pbc_gradient(data, dx, dy, dz):
    grad = np.zeros((3,) + data.shape)
    for axis, d in enumerate([dx, dy, dz]):
        grad[axis] = (np.roll(data, -1, axis=axis) - np.roll(data, 1, axis=axis)) / (2 * d)
    return grad

def pbc_laplacian(data, dx, dy, dz):
    laplacian = np.zeros_like(data)
    for axis, d in enumerate([dx, dy, dz]):
        laplacian += (np.roll(data, -1, axis=axis) + np.roll(data, 1, axis=axis) - 2*data) / (d**2)
    return laplacian

def calculate_ke_density(wannier_data, span_vectors, electron_density):
    A = 1  # ℏ^2/(2m) in A.U.
    
    dx, dy, dz = [np.linalg.norm(v) / (wannier_data.shape[i+1] - 1) for i, v in enumerate(span_vectors)]
    
    laplacian_term = np.zeros_like(electron_density)
    for wf in wannier_data:
        laplacian_wf = pbc_laplacian(wf, dx, dy, dz)
        laplacian_term += np.real(np.conj(wf) * laplacian_wf)
    T = -2 * A * laplacian_term
    
    laplacian_density = pbc_laplacian(electron_density, dx, dy, dz)
    T_corr = 0.5 * A * laplacian_density
    
    grad = pbc_gradient(electron_density, dx, dy, dz)
    grad_density_squared = np.sum(grad**2, axis=0)  # Sum over the first axis (3 components)
    T_bos = 0.25 * A * grad_density_squared / np.maximum(electron_density, 1e-15)
    
    D = T + T_corr - T_bos
    
    return D

def calculate_elf(electron_density, ke_density, normalize=False):
    # Thomas-Fermi kinetic energy density
    tf_ke_density = 3/10 * (3 * np.pi**2)**(2/3) * electron_density**(5/3)
    
    # ELF formula
    elf = 1 / (1 + (ke_density / np.maximum(tf_ke_density, 1e-15))**2)

    if normalize:
        elf = (elf - elf.min()) / (elf.max() - elf.min())

    return elf

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
        elf = calculate_elf(electron_density, ke_density, normalize=True)

        # Write the ELF to a new XSF file
        output_xsf = 'test_elf.xsf'
        write(output_xsf, atoms, format='xsf', data=elf, origin=origin, span_vectors=span_vectors)

        print(f"ELF calculation completed and written to {output_xsf}")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Span vectors:", span_vectors)
        print("Please check that the dimensions of your data match the span vectors.")
