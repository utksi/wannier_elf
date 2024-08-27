import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ase.io.xsf import read_xsf
import mcubes

def plot_elf_slice(elf_data, cell, atoms, atom_types, slice_direction='z', slice_index=None, output_file=None):
    """
    Plot a slice of the ELF data along with atomic positions.
    
    :param elf_data: 3D numpy array of ELF values
    :param cell: Unit cell vectors
    :param atoms: Atomic positions (Cartesian)
    :param atom_types: List of atomic symbols
    :param slice_direction: Direction of the slice ('x', 'y', or 'z')
    :param slice_index: Index of the slice (if None, middle of the grid is used)
    :param output_file: If provided, save the plot to this file
    """
    nx, ny, nz = elf_data.shape
    
    # Convert atomic positions to fractional coordinates
    frac_atoms = cart_to_frac(atoms, cell)
    
    # Create a meshgrid for the chosen slice
    if slice_direction == 'x':
        slice_index = slice_index or nx // 2
        elf_slice = elf_data[slice_index, :, :]
        x, y = np.meshgrid(np.linspace(0, 1, ny), np.linspace(0, 1, nz))
        atom_coords = frac_atoms[:, [1, 2]]
        xlabel, ylabel = 'y', 'z'
    elif slice_direction == 'y':
        slice_index = slice_index or ny // 2
        elf_slice = elf_data[:, slice_index, :]
        x, y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, nz))
        atom_coords = frac_atoms[:, [0, 2]]
        xlabel, ylabel = 'x', 'z'
    else:  # 'z'
        slice_index = slice_index or nz // 2
        elf_slice = elf_data[:, :, slice_index]
        x, y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
        atom_coords = frac_atoms[:, [0, 1]]
        xlabel, ylabel = 'x', 'y'

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ELF
    im = ax.contourf(x, y, elf_slice.T, levels=np.linspace(0, 1, 21), cmap='viridis', extend='both')
    plt.colorbar(im, label='ELF')
    
    # Plot atomic positions
    for coord, atom_type in zip(atom_coords, atom_types):
        ax.plot(coord[0], coord[1], 'ro', markersize=5, label=atom_type)
    
    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title="Atoms")
    
    ax.set_title(f'ELF Slice ({slice_direction}-direction, index {slice_index})')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()
    
    plt.close()

def plot_elf_isosurfaces(elf_data, cell, atoms, atom_types, isovalues=[0.7, 0.8, 0.9], output_file='elf_isosurfaces.html'):
    """Plot ELF isosurfaces along with atomic positions."""
    nx, ny, nz = elf_data.shape
    
    # Convert atomic positions to fractional coordinates
    frac_atoms = cart_to_frac(atoms, cell)
    
    # Create a figure
    fig = go.Figure()
    
    # Add isosurfaces
    for isovalue in isovalues:
        vertices, triangles = mcubes.marching_cubes(elf_data, isovalue)
        
        # Scale vertices to fractional coordinates
        vertices = vertices / np.array([nx, ny, nz])
        
        # Create a mesh3d trace for the isosurface
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            opacity=0.5,
            colorscale=[[0, 'blue'], [1, 'red']],
            intensity=np.ones(len(vertices)) * isovalue,
            intensitymode='cell',
            name=f'ELF = {isovalue}'
        ))
    
    # Add atomic positions
    for atom_type in set(atom_types):
        atom_positions = frac_atoms[np.array(atom_types) == atom_type]
        fig.add_trace(go.Scatter3d(
            x=atom_positions[:, 0],
            y=atom_positions[:, 1],
            z=atom_positions[:, 2],
            mode='markers',
            marker=dict(size=5),
            name=atom_type
        ))
    
    # Set layout
    fig.update_layout(
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z',
            aspectmode='data'
        ),
        title='ELF Isosurfaces with Atomic Positions'
    )
    
    # Save the figure
    fig.write_html(output_file)
    print(f"Interactive plot saved to {output_file}")

def main():
    # Load ELF data
    elf_data, origin, cell, atoms, atom_types = read_xsf('electron_localization_function.xsf')
    
    print(f"ELF data shape: {elf_data.shape}")
    print(f"Number of atoms: {len(atoms)}")
    
    # Plot 2D slices
    for direction in ['x', 'y', 'z']:
        plot_elf_slice(elf_data, cell, atoms, atom_types, 
                       slice_direction=direction, 
                       output_file=f'elf_slice_{direction}.png')
    
    # Plot 3D isosurfaces
    plot_elf_isosurfaces(elf_data, cell, atoms, atom_types)
    
    print("ELF visualization complete.")

if __name__ == "__main__":
    main()