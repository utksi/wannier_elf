import glob
import numpy as np
from ase.io import write
from ase.io.xsf import iread_xsf

def calculate_ke_density(wannier_data, span_vectors, electron_density):
    A = 1  # ℏ^2/(2m) in A.U.
    
    # Calculate grid spacings
    grid_shape = electron_density.shape
    dx, dy, dz = [v[i] / (n - 1) for i, (v, n) in enumerate(zip(span_vectors, grid_shape))]
    
    # Calculate T term
    T = np.zeros_like(electron_density)
    for wf in wannier_data:
        # Calculate Laplacian of wf
        laplacian_wf = np.zeros_like(wf)
        for axis, d in enumerate([dx, dy, dz]):
            laplacian_wf += np.gradient(np.gradient(wf, d, axis=axis), d, axis=axis)
        T += np.real(np.conj(wf) * laplacian_wf)
    T *= -A
    
    # Calculate T_corr term
    laplacian_density = np.zeros_like(electron_density)
    for axis, d in enumerate([dx, dy, dz]):
        laplacian_density += np.gradient(np.gradient(electron_density, d, axis=axis), d, axis=axis)
    T_corr = 0.25 * A * laplacian_density
    
    # Calculate T_bos term
    grad_density = np.array(np.gradient(electron_density, dx, dy, dz))
    grad_density_squared = np.sum(grad_density**2, axis=0)
    T_bos = 0.125 * A * grad_density_squared / np.maximum(electron_density, 1e-15)
    
    D = T + T_corr - T_bos
    
    print(f"T: min={T.min()}, max={T.max()}, mean={T.mean()}")
    print(f"T_corr: min={T_corr.min()}, max={T_corr.max()}, mean={T_corr.mean()}")
    print(f"T_bos: min={T_bos.min()}, max={T_bos.max()}, mean={T_bos.mean()}")
    print(f"D: min={D.min()}, max={D.max()}, mean={D.mean()}")
    
    return D

def calculate_elf(electron_density, ke_density, normalize=True):
    # Thomas-Fermi kinetic energy density
    tf_ke_density = 3/10 * (3 * np.pi**2)**(2/3) * electron_density**(5/3)
    
    # ELF formula
    elf = 1 / (1 + (ke_density / np.maximum(tf_ke_density, 1e-15))**2)
    
    if normalize:
        elf = (elf - elf.min()) / (elf.max() - elf.min())
    
    print(f"ELF: min={elf.min()}, max={elf.max()}, mean={elf.mean()}")
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

    try:
        # Calculate the volume of the unit cell (in Angstrom^3)
        cell_volume = np.abs(np.linalg.det(span_vectors))
        print(f"Unit cell volume: {cell_volume} Å³")

        # Calculate electron density
        electron_density = np.sum(np.abs(wannier_data)**2, axis=0)
        print("Shape of electron_density:", electron_density.shape)
        print(f"Electron density: min={electron_density.min()}, max={electron_density.max()}, mean={electron_density.mean()}")
        
        total_electrons = np.sum(electron_density) * cell_volume / electron_density.size
        print(f"Total electrons: {total_electrons}")

        # Calculate kinetic energy density
        ke_density = calculate_ke_density(wannier_data, span_vectors, electron_density)
        
        # Calculate ELF
        elf = calculate_elf(electron_density, ke_density, normalize=True)

        # Write the ELF to a new XSF file
        output_xsf = 'elf_r.xsf'
        write(output_xsf, atoms, format='xsf', data=elf, origin=origin, span_vectors=span_vectors)

        print(f"ELF calculation completed and written to {output_xsf}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("Span vectors:", span_vectors)
        print("Please check that the dimensions of your data match the span vectors.")