#%%
import numpy as np
import matplotlib.pyplot as plt

def load_elf_data(file_path):
    """Load ELF data from a .npy file."""
    return np.load(file_path)

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

def plot_and_save_slice(slice_data, title, filename):
    """Plot a 2D slice and save it as a PNG file."""
    plt.imshow(slice_data, cmap='viridis', origin='lower')
    plt.colorbar(label='ELF')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    # Save the plot as a PNG file with 300 DPI
    plt.savefig(filename, format='png', dpi=300)
    plt.close()

def main():
    # Load ELF data
    elf_data = load_elf_data('elf.npy')  # Ensure this file exists

    # Determine the shape of the ELF data
    elf_shape = elf_data.shape

    # Example: Choose a plane index for slicing (e.g., middle of each dimension)
    xy_plane_index = elf_shape[2] // 2  # Middle of z-axis
    yz_plane_index = elf_shape[0] // 2  # Middle of x-axis
    zx_plane_index = elf_shape[1] // 2  # Middle of y-axis

    # Extract slices
    xy_slice = extract_slice(elf_data, xy_plane_index, axis=2)
    yz_slice = extract_slice(elf_data, yz_plane_index, axis=0)
    zx_slice = extract_slice(elf_data, zx_plane_index, axis=1)

    # Plot and save slices
    plot_and_save_slice(xy_slice, 'XY Plane Slice', 'xy_plane_slice.png')
    plot_and_save_slice(yz_slice, 'YZ Plane Slice', 'yz_plane_slice.png')
    plot_and_save_slice(zx_slice, 'ZX Plane Slice', 'zx_plane_slice.png')

if __name__ == "__main__":
    main()