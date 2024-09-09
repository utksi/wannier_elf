import glob
import numpy as np
from scipy.fftpack import fftn, ifftn
from scipy.interpolate import interpn
from scipy.ndimage import gaussian_filter
from ase.units import Bohr
from ase.io import write
from ase.io.xsf import iread_xsf
from ase.spacegroup import get_spacegroup
import spglib
from tqdm import tqdm

def periodic_padding(arr, pad_width):
    """Apply periodic padding to a 3D array."""
    return np.pad(arr, pad_width, mode='wrap')

def periodic_gradient(f, dx):
    """Calculate gradient with periodic boundary conditions."""
    pad_width = 1
    padded = periodic_padding(f, pad_width)
    grad = np.zeros((3,) + f.shape)
    for i in range(3):
        grad[i] = (np.roll(padded, -1, axis=i)[pad_width:-pad_width, pad_width:-pad_width, pad_width:-pad_width] 
                   - np.roll(padded, 1, axis=i)[pad_width:-pad_width, pad_width:-pad_width, pad_width:-pad_width]) / (2 * dx[i])
    return grad

def periodic_laplacian(f, dx):
    """Calculate Laplacian with periodic boundary conditions."""
    pad_width = 1
    padded = periodic_padding(f, pad_width)
    laplacian = np.zeros_like(f)
    for i in range(3):
        laplacian += (np.roll(padded, -1, axis=i)[pad_width:-pad_width, pad_width:-pad_width, pad_width:-pad_width] 
                      - 2*padded[pad_width:-pad_width, pad_width:-pad_width, pad_width:-pad_width] 
                      + np.roll(padded, 1, axis=i)[pad_width:-pad_width, pad_width:-pad_width, pad_width:-pad_width]) / dx[i]**2
    return laplacian

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

def calculate_ke_density(wannier_data, span_vectors, electron_density, use_fft=True):
    # Convert span vectors to Bohr (atomic units)
    span_vectors_bohr = span_vectors / Bohr

    # Calculate grid spacings in Bohr
    grid_shape = electron_density.shape
    dx = np.array([v[i] / (n - 1) for i, (v, n) in enumerate(zip(span_vectors_bohr, grid_shape))])
    dV = np.abs(np.linalg.det(span_vectors_bohr)) / np.prod(grid_shape)  # Volume element in Bohr^3

    if use_fft:
        # Generate k-point grid in reciprocal space
        recip_vectors = 2 * np.pi * np.linalg.inv(span_vectors_bohr).T
        kx = np.fft.fftfreq(grid_shape[0]) * grid_shape[0]
        ky = np.fft.fftfreq(grid_shape[1]) * grid_shape[1]
        kz = np.fft.fftfreq(grid_shape[2]) * grid_shape[2]
        kgrid = np.array(np.meshgrid(kx, ky, kz, indexing='ij'))
        k_vectors = np.einsum('ij,jklm->iklm', recip_vectors, kgrid)
        k_squared = np.sum(k_vectors**2, axis=0)

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
        grad_density = np.array([np.real(ifftn(1j * k * density_k)) / dV for k in k_vectors])
        grad_density_squared = np.sum(grad_density**2, axis=0)
        T_bos = 0.125 * grad_density_squared / np.maximum(electron_density, 1e-10)
    else:
        # Calculate T term
        T = np.zeros_like(electron_density)
        for wf in wannier_data:
            laplacian_wf = periodic_laplacian(wf, dx)
            T += np.real(np.conj(wf) * laplacian_wf)
        T *= -0.5  # Factor of -1/2 in atomic units

        # Calculate T_corr term
        laplacian_density = periodic_laplacian(electron_density, dx)
        T_corr = 0.125 * laplacian_density

        # Calculate T_bos term
        grad_density = periodic_gradient(electron_density, dx)
        grad_density_squared = np.sum(grad_density**2, axis=0)
        T_bos = 0.125 * grad_density_squared / np.maximum(electron_density, 1e-10)

    # Total kinetic energy density
    D = T + T_corr - T_bos

    # Apply Gaussian smoothing to reduce noise
    D = gaussian_filter(D, sigma=1)

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

    print(f"tf_ke_density: min={tf_ke_density.min()}, max={tf_ke_density.max()}, mean={tf_ke_density.mean()}")
    print(f"chi: min={chi.min()}, max={chi.max()}, mean={chi.mean()}")
    print(f"ELF (before normalization): min={elf.min()}, max={elf.max()}, mean={elf.mean()}")

    if normalize:
        elf = (elf - elf.min()) / (elf.max() - elf.min())
        print(f"ELF (after normalization): min={elf.min()}, max={elf.max()}, mean={elf.mean()}")

    return elf

def interpolate_3d(data, old_grid, new_grid):
    """Interpolate 3D data to a new grid."""
    points = tuple(map(np.ravel, np.meshgrid(*new_grid, indexing='ij')))
    return interpn(old_grid, data, points).reshape(len(new_grid[0]), len(new_grid[1]), len(new_grid[2]))

def mask_boundaries(data, mask_width):
    """Mask out boundary regions of the data."""
    masked = data.copy()
    masked[:mask_width, :, :] = 0
    masked[-mask_width:, :, :] = 0
    masked[:, :mask_width, :] = 0
    masked[:, -mask_width:, :] = 0
    masked[:, :, :mask_width] = 0
    masked[:, :, -mask_width:] = 0
    return masked

# Main script
if __name__ == "__main__":
    # Get all Wannier function XSF files
    wannier_files = glob.glob('./wannier_*.xsf')
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
                print(f" Data shape: {data.shape}")
            except Exception as e:
                print(f" Error reading file {file}: {e}")

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
            symmetrize_input = input("Symmetrize electron density? (y/n): ").lower()
            if symmetrize_input in ['y', 'n']:
                symmetrize = symmetrize_input == 'y'
                break
            print("Invalid input. Please enter 'y' or 'n'.")

        if symmetrize:
            # Get spacegroup
            spacegroup = get_spacegroup(atoms)
            print(f"Detected spacegroup: {spacegroup.symbol}")
            
            # Symmetrize electron density
            electron_density = symmetrize_density(electron_density, atoms)
            print("Electron density symmetrized according to crystal symmetry.")
            print(f"Symmetrized electron density: min={electron_density.min()}, max={electron_density.max()}, mean={electron_density.mean()}")

        # Ask user for calculation method preference
        while True:
            method_input = input("Use FFT method? (y/n): ").lower()
            if method_input in ['y', 'n']:
                use_fft = method_input == 'y'
                break
            print("Invalid input. Please enter 'y' or 'n'.")

        # Calculate kinetic energy density
        ke_density = calculate_ke_density(wannier_data, span_vectors, electron_density, use_fft=use_fft)

        # Ask user for normalization preference
        while True:
            normalize_input = input("Normalize ELF? (y/n): ").lower()
            if normalize_input in ['y', 'n']:
                normalize = normalize_input == 'y'
                break
            print("Invalid input. Please enter 'y' or 'n'.")

        # Ask user for interpolation preference
        while True:
            interpolate_input = input("Interpolate ELF? (y/n): ").lower()
            if interpolate_input in ['y', 'n']:
                interpolate = interpolate_input == 'y'
                break
            print("Invalid input. Please enter 'y' or 'n'.")

        # Ask user for masking preference
        while True:
            mask_input = input("Mask ELF boundaries? (y/n): ").lower()
            if mask_input in ['y', 'n']:
                mask = mask_input == 'y'
                break
            print("Invalid input. Please enter 'y' or 'n'.")

        # Calculate ELF
        elf = calculate_elf(electron_density, ke_density, normalize=normalize)

        if mask:
            mask_width = 2  # Adjust as needed
            elf = mask_boundaries(elf, mask_width)
            print(f"ELF after masking: min={elf.min()}, max={elf.max()}, mean={elf.mean()}")

        # Determine output filename based on user choices
        output_xsf = f"elf_{'fft' if use_fft else 'real'}_{('sym' if symmetrize else 'nosym')}_{('norm' if normalize else 'nonorm')}_{('interp' if interpolate else 'nointerp')}_{('mask' if mask else 'nomask')}.xsf"

        if interpolate:
            # Perform interpolation
            old_shape = elf.shape
            new_shape = tuple(2*x - 1 for x in old_shape)  # Double the grid density
            
            # Create old and new grids (normalized coordinates)
            old_grid = tuple(np.linspace(0, 1, n) for n in old_shape)
            new_grid = tuple(np.linspace(0, 1, n) for n in new_shape)
            
            elf_interpolated = interpolate_3d(elf, old_grid, new_grid)
            print(f"Interpolated ELF shape: {elf_interpolated.shape}")
            
            # Write interpolated ELF to XSF file
            write(output_xsf, atoms, format='xsf', data=elf_interpolated, origin=origin, span_vectors=span_vectors)
        else:
            # Write non-interpolated ELF to XSF file
            write(output_xsf, atoms, format='xsf', data=elf, origin=origin, span_vectors=span_vectors)

        print(f"ELF calculation completed and written to {output_xsf}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("Span vectors:", span_vectors)
        print("Please check that the dimensions of your data match the span vectors.")