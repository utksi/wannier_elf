import glob
import numpy as np
from scipy.fftpack import fftn, ifftn
from ase.units import Bohr
from ase.io import write
from ase.io.xsf import iread_xsf
from ase.spacegroup import get_spacegroup
import spglib
from tqdm import tqdm

def symmetrize_density(density, atoms, symprec=1e-5):
    """Symmetrize the charge density based on the crystal symmetry."""
    cell = (atoms.get_cell(), atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    symmetry = spglib.get_symmetry(cell, symprec=symprec)
    rotations = symmetry['rotations']
    translations = symmetry['translations']

    density_sym = np.zeros_like(density)
    count = np.zeros_like(density, dtype=int)

    x, y, z = np.meshgrid(np.arange(density.shape[0]),
                          np.arange(density.shape[1]),
                          np.arange(density.shape[2]), indexing='ij')
    positions = np.stack([x, y, z], axis=-1)

    for rot, trans in tqdm(zip(rotations, translations), total=len(rotations), desc="Symmetrizing density"):
        new_pos = np.einsum('ij,klmj->klmi', rot, positions) + trans * density.shape
        new_pos = np.round(new_pos).astype(int) % density.shape
        density_sym[new_pos[..., 0], new_pos[..., 1], new_pos[..., 2]] += density
        count[new_pos[..., 0], new_pos[..., 1], new_pos[..., 2]] += 1

    density_sym = np.where(count > 0, density_sym / count, density)
    return density_sym

def calculate_ke_density(wannier_data, span_vectors, electron_density):
    # Convert span vectors to Bohr (atomic units)
    span_vectors_bohr = span_vectors / Bohr

    # Calculate grid spacings in Bohr
    grid_shape = electron_density.shape
    dx, dy, dz = [v[i] / (n - 1) for i, (v, n) in enumerate(zip(span_vectors_bohr, grid_shape))]
    dV = dx * dy * dz  # Volume element in Bohr^3

    # Generate k-point grid in reciprocal space
    kx = 2 * np.pi * np.fft.fftfreq(grid_shape[0], dx)
    ky = 2 * np.pi * np.fft.fftfreq(grid_shape[1], dy)
    kz = 2 * np.pi * np.fft.fftfreq(grid_shape[2], dz)
    kgrid = np.array(np.meshgrid(kx, ky, kz, indexing='ij'))
    k_squared = np.sum(kgrid**2, axis=0)

    # Calculate T term (kinetic energy density)
    T = np.zeros_like(electron_density)
    for wf in wannier_data:
        wf_k = fftn(wf) * dV  # FFT of wavefunction
        T += np.real(ifftn(k_squared * np.abs(wf_k)**2)) / dV
    T *= 0.5  # Factor of 1/2 in atomic units

    # Calculate T_corr term (correction to kinetic energy density)
    density_k = fftn(electron_density) * dV
    T_corr = 0.125 * np.real(ifftn(k_squared * density_k)) / dV

    # Calculate T_bos term (Bose part of kinetic energy density)
    grad_density = np.array([np.real(ifftn(1j * k * density_k)) / dV for k in kgrid])
    grad_density_squared = np.sum(grad_density**2, axis=0)
    T_bos = 0.125 * grad_density_squared / np.maximum(electron_density, 1e-10)

    # Total kinetic energy density
    D = T + T_corr - T_bos

    print(f"T: min={T.min()}, max={T.max()}, mean={T.mean()}")
    print(f"T_corr: min={T_corr.min()}, max={T_corr.max()}, mean={T_corr.mean()}")
    print(f"T_bos: min={T_bos.min()}, max={T_bos.max()}, mean={T_bos.mean()}")
    print(f"D: min={D.min()}, max={D.max()}, mean={D.mean()}")

    return D

def calculate_elf(electron_density, ke_density, normalize=False):
    # Thomas-Fermi kinetic energy density (in atomic units)
    tf_ke_density = 3/10 * (3 * np.pi**2)**(2/3) * electron_density**(5/3)

    # ELF formula
    chi = ke_density / np.maximum(tf_ke_density, 1e-10)
    elf = 1 / (1 + chi**2)

    if normalize:
        elf = (elf - elf.min()) / (elf.max() - elf.min())

    print(f"tf_ke_density: min={tf_ke_density.min()}, max={tf_ke_density.max()}, mean={tf_ke_density.mean()}")
    print(f"chi: min={chi.min()}, max={chi.max()}, mean={chi.mean()}")
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

        # Ask user for symmetrization preference
        while True:
            symmetrize_input = input("Symmetrize charge density? (y/n): ").lower()
            if symmetrize_input in ['y', 'n']:
                symmetrize = symmetrize_input == 'y'
                break
            print("Invalid input. Please enter 'y' or 'n'.")

        if symmetrize:
            # Get spacegroup
            spacegroup = get_spacegroup(atoms)
            print(f"Detected spacegroup: {spacegroup.symbol}")
            
            # Symmetrize charge density
            electron_density = symmetrize_density(electron_density, atoms)
            print("Charge density symmetrized according to crystal symmetry.")

        # Calculate kinetic energy density
        ke_density = calculate_ke_density(wannier_data, span_vectors, electron_density)

        # Ask user for normalization preference
        while True:
            normalize_input = input("Normalize ELF? (y/n): ").lower()
            if normalize_input in ['y', 'n']:
                normalize = normalize_input == 'y'
                break
            print("Invalid input. Please enter 'y' or 'n'.")

        # Calculate ELF
        elf = calculate_elf(electron_density, ke_density, normalize=normalize)

        # Determine output filename based on user choices
        output_xsf = f"elf_{'sym' if symmetrize else 'nosym'}_{'norm' if normalize else 'nonorm'}.xsf"

        # Write ELF to XSF file
        write(output_xsf, atoms, format='xsf', data=elf, origin=origin, span_vectors=span_vectors)

        print(f"ELF calculation completed and written to {output_xsf}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("Span vectors:", span_vectors)
        print("Please check that the dimensions of your data match the span vectors.")