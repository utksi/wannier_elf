import glob
import numpy as np
from ase.io import write
from ase.io.xsf import iread_xsf

def calculate_ke_density(wannier_data, span_vectors, electron_density):
    A = 1  # ℏ^2/(2m) but in A.U.
    
    # Calculate grid spacings from span vectors
    dx = np.linalg.norm(span_vectors[0]) / (wannier_data.shape[1] - 1)
    dy = np.linalg.norm(span_vectors[1]) / (wannier_data.shape[2] - 1)
    dz = np.linalg.norm(span_vectors[2]) / (wannier_data.shape[3] - 1)
    
    # First term: −2A∑ψ^{*}∇^2 ψ
    laplacian_term = np.zeros_like(electron_density)
    for wf in wannier_data:
        grad_x = np.gradient(wf, dx, axis=0)
        grad_y = np.gradient(wf, dy, axis=1)
        grad_z = np.gradient(wf, dz, axis=2)
        laplacian_wf = (np.gradient(grad_x, dx, axis=0) +
                        np.gradient(grad_y, dy, axis=1) +
                        np.gradient(grad_z, dz, axis=2))
        laplacian_term += np.real(np.conj(wf) * laplacian_wf)
    T = -2 * A * laplacian_term
    
    # Second term: (A/2) ∇^2 n
    grad_x = np.gradient(electron_density, dx, axis=0)
    grad_y = np.gradient(electron_density, dy, axis=1)
    grad_z = np.gradient(electron_density, dz, axis=2)
    laplacian_density = (np.gradient(grad_x, dx, axis=0) +
                         np.gradient(grad_y, dy, axis=1) +
                         np.gradient(grad_z, dz, axis=2))
    T_corr = 0.5 * A * laplacian_density
    
    # Third term: [(A/(4n)] (∇n)^2
    grad_density_squared = grad_x**2 + grad_y**2 + grad_z**2
    T_bos = 0.25 * A * grad_density_squared / np.maximum(electron_density, 1e-12)
    
    # Combine terms as per CASTEP/VASP formula: D = T + T_corr - T_bos
    D = T + T_corr - T_bos
    
    return D

def calculate_elf(electron_density, ke_density, normalize):
    # Thomas-Fermi kinetic energy density
    tf_ke_density = 3/10 * (3 * np.pi**2)**(2/3) * electron_density**(5/3)
    
    # ELF formula
    elf = 1 / (1 + (ke_density / np.maximum(tf_ke_density, 1e-12))**2)

    if normalize:
        elf = (elf - elf.min())/(elf.max() - elf.min())

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
        elf = calculate_elf(electron_density, ke_density, normalize = False)

        # Write the ELF to a new XSF file
        output_xsf = 'nopbc_elf.xsf'
        write(output_xsf, atoms, format='xsf', data=elf, origin=origin, span_vectors=span_vectors)

        print(f"ELF calculation completed and written to {output_xsf}")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Span vectors:", span_vectors)
        print("Please check that the dimensions of your data match the span vectors.")
