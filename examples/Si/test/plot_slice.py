#%%
import matplotlib.pyplot as plt
from ase.io.xsf import read_xsf

def read_elf_data_xsf(file_path):
    """Read ELF data and metadata from an XSF file."""
    with open(file_path, 'r') as f:
        data, origin, span_vectors, atoms = read_xsf(f, read_data=True)
    return data, origin, span_vectors, atoms

def extract_slice(elf_data, plane_index, axis=0):
    """Extract a 2D slice from 3D ELF data."""
    if axis == 0:
        return elf_data[plane_index, :, :]  # yz plane
    elif axis == 1:
        return elf_data[:, plane_index, :]  # xz plane
    elif axis == 2:
        return elf_data[:, :, plane_index]  # xy plane
    else:
        raise ValueError('Invalid axis for slicing')

def plot_and_save_slice(slice_data, title, filename, extent, axis_labels):
    """Plot a 2D slice and save it as a PNG file."""
    plt.imshow(slice_data, cmap='viridis', origin='lower', extent=extent)
    plt.colorbar(label='ELF')
    plt.title(title)
    plt.xlabel(axis_labels[0] + ' (Å)')
    plt.ylabel(axis_labels[1] + ' (Å)')
    # Save the plot as a PNG file with 300 DPI
    plt.savefig(filename, format='png', dpi=300)
    plt.close()

def main():
    # Load ELF data and metadata from the XSF file
    elf_data, origin, span_vectors, atoms = read_elf_data_xsf('fourier_elf.xsf')  # Ensure this file exists

    # Print the lattice vectors
    print("Lattice Vectors (span vectors):")
    print(span_vectors)

    # Determine the shape of the ELF data
    elf_shape = elf_data.shape

    # Example: Choose a plane index for slicing (e.g., middle of each dimension)
    xy_plane_index = elf_shape[2] // 4  # Middle of z-axis
    yz_plane_index = elf_shape[0] // 4  # Middle of x-axis
    zx_plane_index = elf_shape[1] // 4  # Middle of y-axis

    # Extract slices
    xy_slice = extract_slice(elf_data, xy_plane_index, axis=2)
    yz_slice = extract_slice(elf_data, yz_plane_index, axis=0)
    zx_slice = extract_slice(elf_data, zx_plane_index, axis=1)

    # Plot and save slices with correct extents
    plot_and_save_slice(xy_slice, 'XY Plane Slice', 'xy_plane_slice.png', [0, span_vectors[0, 0], 0, span_vectors[1, 1]], ['X', 'Y'])
    plot_and_save_slice(yz_slice, 'YZ Plane Slice', 'yz_plane_slice.png', [0, span_vectors[1, 1], 0, span_vectors[2, 2]], ['Y', 'Z'])
    plot_and_save_slice(zx_slice, 'ZX Plane Slice', 'zx_plane_slice.png', [0, span_vectors[2, 2], 0, span_vectors[0, 0]], ['Z', 'X'])

if __name__ == "__main__":
    main()
