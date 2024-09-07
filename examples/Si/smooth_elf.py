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

def calculate_elf(electron_density, ke_density, normalize=False, threshold=1e-6, smooth=False, sigma=0.5):
    # Thomas-Fermi kinetic energy density
    tf_ke_density = 3/10 * (3 * np.pi**2)**(2/3) * electron_density**(5/3)
    
    # ELF formula
    elf = 1 / (1 + (ke_density / np.maximum(tf_ke_density, 1e-12))**2)
    
    # Apply threshold to filter out very small values
    elf[elf < threshold] = 0
    
    if smooth:
        elf = ndimage.gaussian_filter(elf, sigma=sigma)
    
    if normalize:
        elf = (elf - elf.min()) / (elf.max() - elf.min())
    
    return elf

def analyze_elf(elf, threshold=0.9, min_volume=5):
    high_elf = elf > threshold
    labeled, num_features = ndimage.label(high_elf)
    
    print(f"Number of high-ELF regions (>{threshold}): {num_features}")
    
    significant_regions = 0
    for i in range(1, num_features + 1):
        region = labeled == i
        volume = region.sum()
        if volume >= min_volume:
            significant_regions += 1
            print(f"Region {i}: Volume = {volume} grid points")
    
    print(f"Number of significant high-ELF regions: {significant_regions}")
    return labeled, significant_regions

def analyze_elf_positions(elf, atoms, labeled, origin, span_vectors, threshold=0.9, bond_cutoff=2.5):
    bond_regions = 0
    interstitial_regions = 0
    for i in range(1, labeled.max() + 1):
        region = labeled == i
        if region.sum() >= 5:  # Consider only significant regions
            region_indices = np.argwhere(region)
            region_center = region_indices.mean(axis=0)
            
            # Convert grid coordinates to real-space coordinates
            real_center = origin + np.dot(region_center, span_vectors)
            
            # Find nearest atoms
            distances = np.linalg.norm(atoms.positions - real_center, axis=1)
            nearest_atoms = np.argsort(distances)[:2]
            
            if distances[nearest_atoms[0]] < bond_cutoff:
                if distances[nearest_atoms[1]] < bond_cutoff:
                    bond_regions += 1
                    print(f"Region {i}: Bond between atoms {nearest_atoms[0]} and {nearest_atoms[1]}")
                else:
                    print(f"Region {i}: Near atom {nearest_atoms[0]}")
            else:
                interstitial_regions += 1
                print(f"Region {i}: Interstitial, nearest atom {nearest_atoms[0]} at distance {distances[nearest_atoms[0]]:.3f}")
    
    print(f"Bond regions: {bond_regions}")
    print(f"Interstitial regions: {interstitial_regions}")

def check_elf_symmetry(elf, atoms, threshold=0.9):
    from ase.spacegroup import get_spacegroup
    sg = get_spacegroup(atoms)
    
    high_elf = elf > threshold
    
    # Get the fractional coordinates of high ELF points
    high_elf_points = np.argwhere(high_elf)
    frac_coords = np.dot(high_elf_points, np.linalg.inv(atoms.cell))
    
    symmetry_consistent = True
    for op in sg.get_symop():
        # Symmetry operations are returned as (rotation, translation) tuples
        rotation, translation = op
        
        rotated_coords = np.dot(frac_coords, rotation.T) + translation
        
        # Wrap coordinates back into the unit cell
        rotated_coords = rotated_coords % 1.0
        
        # Convert back to grid indices
        rotated_indices = np.round(np.dot(rotated_coords, atoms.cell)).astype(int)
        
        # Check if the rotated points are also high ELF regions
        for idx in rotated_indices:
            if not np.all((idx >= 0) & (idx < elf.shape)):
                continue  # Skip points outside the grid
            if not high_elf[tuple(idx)]:
                symmetry_consistent = False
                break
        
        if not symmetry_consistent:
            break
    
    print(f"ELF regions {'are' if symmetry_consistent else 'are not'} consistent with crystal symmetry")

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
        elf = calculate_elf(electron_density, ke_density, normalize=False, threshold=1e-6, smooth=True, sigma=0.5)

        # Analyze high-ELF regions
        labeled, significant_regions = analyze_elf(elf, threshold=0.9, min_volume=5)
        
        # Analyze positions of high-ELF regions
        analyze_elf_positions(elf, atoms, labeled, origin, span_vectors, threshold=0.9, bond_cutoff=2.5)
        
        # Check ELF symmetry
        check_elf_symmetry(elf, atoms, threshold=0.9)

        # Write the ELF to a new XSF file
        output_xsf = 'smooth_elf.xsf'
        write(output_xsf, atoms, format='xsf', data=elf, origin=origin, span_vectors=span_vectors)

        # Save significant high-ELF regions for visualization
        significant_high_elf = (labeled > 0).astype(float)
        write('significant_high_elf.xsf', atoms, format='xsf', data=significant_high_elf, origin=origin, span_vectors=span_vectors)

        print(f"ELF calculation completed and written to {output_xsf}")
        print("Significant high-ELF regions written to significant_high_elf.xsf")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Span vectors:", span_vectors)
        print("Please check that the dimensions of your data match the span vectors.")
