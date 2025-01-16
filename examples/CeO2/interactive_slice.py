# %%
import plotly.graph_objects as go
from ase.io.xsf import read_xsf
from scipy.ndimage import gaussian_filter


def read_elf_data_xsf(file_path):
    """Read ELF data and metadata from an XSF file."""
    with open(file_path, "r") as f:
        data, origin, span_vectors, atoms = read_xsf(f, read_data=True)
    return data, origin, span_vectors, atoms


def smooth_elf_data(elf_data, sigma=1.0):
    """Apply Gaussian smoothing to ELF data."""
    return gaussian_filter(elf_data, sigma=sigma)


def create_interactive_slice_plot(elf_data, span_vectors, output_file="elf_plot.html"):
    """Create an interactive plot with sliders for slicing the ELF data and save as HTML."""
    elf_shape = elf_data.shape
    fig = go.Figure()

    # Add XY slices
    for z in range(elf_shape[2]):
        fig.add_trace(
            go.Heatmap(
                z=elf_data[:, :, z],
                colorscale="Viridis",
                visible=(z == 0),
                showscale=False,
                name=f"XY Slice {z} (z={z})",
                x0=0,
                dx=span_vectors[0, 0] / elf_shape[0],
                y0=0,
                dy=span_vectors[1, 1] / elf_shape[1],
            )
        )

    # Add YZ slices
    for x in range(elf_shape[0]):
        fig.add_trace(
            go.Heatmap(
                z=elf_data[x, :, :],
                colorscale="Viridis",
                visible=False,
                showscale=False,
                name=f"YZ Slice {x} (x={x})",
                x0=0,
                dx=span_vectors[1, 1] / elf_shape[1],
                y0=0,
                dy=span_vectors[2, 2] / elf_shape[2],
            )
        )

    # Add ZX slices
    for y in range(elf_shape[1]):
        fig.add_trace(
            go.Heatmap(
                z=elf_data[:, y, :],
                colorscale="Viridis",
                visible=False,
                showscale=False,
                name=f"ZX Slice {y} (y={y})",
                x0=0,
                dx=span_vectors[2, 2] / elf_shape[2],
                y0=0,
                dy=span_vectors[0, 0] / elf_shape[0],
            )
        )

    # Create sliders for each plane
    sliders = []

    # XY plane slider
    xy_steps = []
    for i in range(elf_shape[2]):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}, {"title": f"XY Slice {i}"}],
        )
        step["args"][0]["visible"][i] = True  # Toggle i-th trace to be visible
        xy_steps.append(step)

    sliders.append(
        dict(
            active=0,
            currentvalue={"prefix": "XY Slice: "},
            pad={"t": 50},
            steps=xy_steps,
        )
    )

    # YZ plane slider
    yz_steps = []
    for i in range(elf_shape[0]):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}, {"title": f"YZ Slice {i}"}],
        )
        step["args"][0]["visible"][elf_shape[2] + i] = (
            True  # Toggle i-th trace to be visible
        )
        yz_steps.append(step)

    sliders.append(
        dict(
            active=0,
            currentvalue={"prefix": "YZ Slice: "},
            pad={"t": 50},
            steps=yz_steps,
        )
    )

    # ZX plane slider
    zx_steps = []
    for i in range(elf_shape[1]):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}, {"title": f"ZX Slice {i}"}],
        )
        step["args"][0]["visible"][elf_shape[2] + elf_shape[0] + i] = (
            True  # Toggle i-th trace to be visible
        )
        zx_steps.append(step)

    sliders.append(
        dict(
            active=0,
            currentvalue={"prefix": "ZX Slice: "},
            pad={"t": 50},
            steps=zx_steps,
        )
    )

    # Add buttons to switch between sliders and slices
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[
                            {"visible": [False] * len(fig.data)},
                            {"sliders": [sliders[0]]},
                        ],
                        label="XY Plane",
                        method="update",
                    ),
                    dict(
                        args=[
                            {"visible": [False] * len(fig.data)},
                            {"sliders": [sliders[1]]},
                        ],
                        label="YZ Plane",
                        method="update",
                    ),
                    dict(
                        args=[
                            {"visible": [False] * len(fig.data)},
                            {"sliders": [sliders[2]]},
                        ],
                        label="ZX Plane",
                        method="update",
                    ),
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top",
            )
        ],
        sliders=sliders,
        title="Interactive ELF Slices",
        xaxis_title="X (Å)",
        yaxis_title="Y (Å)",
    )

    # Save the plot to an HTML file
    fig.write_html(output_file)
    print(f"Plot saved to {output_file}")

    # Show the plot
    fig.show()


def main():
    # Load ELF data and metadata from the XSF file
    elf_data, origin, span_vectors, atoms = read_elf_data_xsf(
        "wan_elf.xsf"
    )  # Ensure this file exists

    # Apply smoothing to ELF data
    smooth_sigma = 1.0  # Adjust as needed
    elf_data_smoothed = smooth_elf_data(elf_data, sigma=smooth_sigma)

    # Print the lattice vectors
    print("Lattice Vectors (span vectors):")
    print(span_vectors)

    # Create an interactive plot
    create_interactive_slice_plot(
        elf_data_smoothed, span_vectors, output_file="elf_plot.html"
    )


if __name__ == "__main__":
    main()
