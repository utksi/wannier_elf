import warnings
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from enum import Enum
from scipy.interpolate import interpn
from scipy.ndimage import gaussian_filter
from scipy.fft import fftn, ifftn
import ase.io
from ase.io import write
from ase.io.xsf import iread_xsf
from ase.spacegroup import get_spacegroup
import spglib

# Suppress spglib deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="spglib")


class GradientMethod(Enum):
    FINITE_DIFF = "finite_diff"
    FOURIER = "fourier"
    SOBEL = "sobel"


class SymmetrizationMethod(Enum):
    REAL = "real"  # Real space average symmetrization
    RECIPROCAL = "reciprocal"  # Reciprocal space symmetrization like VASP


class WannierELFCalculator:
    def __init__(
        self,
        input_files: List[Path],
        gradient_method: GradientMethod = GradientMethod.FINITE_DIFF,
        symmetrization_method: SymmetrizationMethod = SymmetrizationMethod.REAL,
    ):
        """
        Initialize calculator with Wannier function XSF files

        Args:
            input_files: List of paths to Wannier XSF files
            gradient_method: Method to use for gradient calculation
            symmetrization_method: Method to use for symmetrization
        """
        self.gradient_method = gradient_method
        self.symmetrization_method = symmetrization_method
        self.logger = logging.getLogger(__name__)

        # Read structure and Wannier data from XSF files
        self.atoms, self.wannier_data, self.origin, self.span_vectors = (
            self.read_wannier_files(input_files)
        )
        self.logger.info(f"Read {len(self.wannier_data)} Wannier functions")

        # Get symmetry information using spglib
        cell = (self.atoms.cell, self.atoms.get_scaled_positions(), self.atoms.numbers)
        self.spacegroup = get_spacegroup(self.atoms)
        self.symmetry = spglib.get_symmetry_dataset(cell)
        self.logger.info(f"Detected space group: {self.spacegroup.symbol}")

        # Store rotations and translations
        self.rotations = self.symmetry["rotations"]
        self.translations = self.symmetry["translations"]

        # Store atomic environment information
        self.atomic_info = self._compute_atomic_environment()

    def _compute_atomic_environment(self) -> Dict:
        """
        Compute and store information about the atomic environment
        Returns dictionary with atomic positions in various coordinates
        """
        nx, ny, nz = self.wannier_data[0].shape

        # Get atomic positions in various coordinates
        cart_positions = self.atoms.get_positions()  # Cartesian coordinates (Å)
        scaled_positions = self.atoms.get_scaled_positions()  # Fractional coordinates
        grid_positions = (scaled_positions * [nx, ny, nz]) % np.array([nx, ny, nz])

        return {
            "symbols": self.atoms.get_chemical_symbols(),
            "cart_positions": cart_positions,
            "scaled_positions": scaled_positions,
            "grid_positions": grid_positions,
            "atomic_numbers": self.atoms.numbers,
        }

    def read_wannier_files(
        self, input_files: List[Path]
    ) -> Tuple[ase.Atoms, List[np.ndarray], np.ndarray, np.ndarray]:
        """
        Read Wannier function data from XSF files using ASE.
        Note: Data is read in grid coordinates (fractional).

        Args:
            input_files: List of paths to Wannier XSF files
        Returns:
            atoms: ASE Atoms object containing the structure
            wannier_data: List of 3D numpy arrays containing Wannier functions
            origin: Origin of the grid
            span_vectors: Cell vectors for the grid
        """
        wannier_data = []
        atoms = None
        origin = None
        span_vectors = None

        for file in input_files:
            self.logger.info(f"Reading file: {file}")
            with open(file, "r") as fileobj:
                try:
                    images = list(iread_xsf(fileobj, read_data=True))
                    data, origin, span_vectors = images[-1]
                    if atoms is None:
                        atoms = images[0]
                    wannier_data.append(data)
                    self.logger.info(f" Data shape: {data.shape}")
                except Exception as e:
                    self.logger.error(f" Error reading file {file}: {e}")
                    raise

            # Validate consistency if this isn't the first file
            if len(wannier_data) > 1:
                if wannier_data[-1].shape != wannier_data[0].shape:
                    raise ValueError(f"Inconsistent grid dimensions in {file}")

        return atoms, wannier_data, origin, span_vectors

    def compute_gradient(self, data: np.ndarray) -> np.ndarray:
        """
        Compute gradient considering the non-orthogonal cell geometry.
        Input data is on a grid in fractional coordinates.

        Args:
            data: 3D numpy array of values on grid points

        Returns:
            gradient: numpy array with shape (nx, ny, nz, 3)
        """
        # Get grid dimensions
        nx, ny, nz = data.shape
        grid_spacings = [1.0 / nx, 1.0 / ny, 1.0 / nz]  # Fractional grid spacings

        # Compute gradient in fractional coordinates
        grad_frac = np.empty(data.shape + (3,))
        grad_frac[..., 0], grad_frac[..., 1], grad_frac[..., 2] = np.gradient(
            data, grid_spacings[0], grid_spacings[1], grid_spacings[2], edge_order=2
        )

        # Transform to Cartesian coordinates
        cell = self.atoms.get_cell()
        inv_cell_T = np.linalg.inv(cell).T

        # Reshape gradient for matrix multiplication
        grad_frac_reshaped = grad_frac.reshape(-1, 3).T

        # Transform to Cartesian coordinates
        grad_cartesian = inv_cell_T @ grad_frac_reshaped

        # Reshape back to original grid shape
        grad_cartesian = grad_cartesian.T.reshape(data.shape + (3,))

        return grad_cartesian

    def _compute_distance_to_atoms(
        self, point: np.ndarray, grid_shape: Tuple[int, int, int]
    ) -> float:
        """
        Compute minimum distance from a point to any atom, considering periodicity.

        Args:
            point: Point in grid coordinates
            grid_shape: Shape of the grid (nx, ny, nz)

        Returns:
            Minimum distance to any atom in grid units
        """
        nx, ny, nz = grid_shape
        min_dist = float("inf")

        # Convert point to fractional coordinates
        point_frac = point / np.array([nx, ny, nz])

        # Get atomic positions
        for atom_pos in self.atomic_info["scaled_positions"]:
            # Calculate difference vector considering periodicity
            diff = point_frac - atom_pos
            diff = diff - np.round(diff)  # Apply minimum image convention

            # Convert to Cartesian coordinates for proper distance calculation
            cart_diff = diff @ self.atoms.get_cell()
            dist = np.linalg.norm(cart_diff)
            min_dist = min(min_dist, dist)

        return min_dist

    def _get_reciprocal_vectors(self) -> np.ndarray:
        """
        Get reciprocal lattice vectors (2π * inverse transpose of real space lattice vectors).
        """
        return 2 * np.pi * np.linalg.inv(self.atoms.get_cell()).T

    def _to_reciprocal_space(self, field: np.ndarray) -> np.ndarray:
        """
        Transform field to reciprocal space using FFT.
        """
        return fftn(field) / np.prod(field.shape[:3])

    def _to_real_space(self, field: np.ndarray) -> np.ndarray:
        """
        Transform field to real space using inverse FFT.
        """
        return ifftn(field).real * np.prod(field.shape[:3])

    def symmetrize_field(self, field: np.ndarray, field_name: str) -> np.ndarray:
        """
        Symmetrize a field according to crystal symmetry.
        Uses either real space or reciprocal space symmetrization based on initialization.

        Args:
            field: Scalar field with shape (nx, ny, nz) or vector field with shape (nx, ny, nz, 3)
            field_name: Name of field for logging

        Returns:
            symmetrized_field: Field with same shape as input but symmetrized
        """
        if self.symmetrization_method == SymmetrizationMethod.RECIPROCAL:
            return self._reciprocal_symmetrize(field, field_name)
        else:
            return self._real_symmetrize(field, field_name)

    def _real_symmetrize(self, field: np.ndarray, field_name: str) -> np.ndarray:
        """
        Traditional symmetrization using averaging of symmetry-equivalent points in real space.
        """
        self.logger.info(f"Starting real-space symmetrization of {field_name}")

        # Store original field for validation
        original_field = field.copy()

        # Get the shape and dimensions
        field_shape = field.shape
        spatial_dims = field_shape[:3]
        component_dims = field_shape[3:]  # Empty for scalar field, (3,) for vector
        nx, ny, nz = spatial_dims
        n_ops = len(self.rotations)

        # Create fractional grid coordinates
        grid_points = np.indices(spatial_dims).reshape(3, -1).T / np.array([nx, ny, nz])

        # Initialize array to accumulate field values
        num_points = len(grid_points)
        if component_dims:
            sym_field_values = np.zeros((num_points, n_ops) + component_dims)
        else:
            sym_field_values = np.zeros((num_points, n_ops))

        # Grid for interpolation
        grid = (np.arange(nx), np.arange(ny), np.arange(nz))

        # Reshape field for interpolation
        field_reshaped = field.reshape(spatial_dims + (-1,))

        # Apply symmetry operations
        for i, (rot, trans) in enumerate(zip(self.rotations, self.translations)):
            if (i + 1) % 10 == 0 or i == n_ops - 1:
                self.logger.info(f"Processing symmetry operation {i+1}/{n_ops}")

            # Apply rotation and translation to fractional coordinates
            transformed_coords = (grid_points @ rot.T + trans) % 1.0

            # Convert fractional coordinates to grid indices
            transformed_indices = transformed_coords * np.array([nx, ny, nz])

            # Ensure indices are within the grid
            transformed_indices %= np.array([nx, ny, nz])

            # Interpolate field values at transformed positions
            field_values = []
            for comp in range(field_reshaped.shape[-1]):
                values = interpn(
                    grid,
                    field_reshaped[..., comp],
                    transformed_indices,
                    method="linear",
                    bounds_error=False,
                    fill_value=0.0,
                )
                field_values.append(values)

            # Stack component values
            field_values = np.stack(field_values, axis=-1)

            # If scalar field, squeeze the last dimension
            if field_values.shape[-1] == 1:
                field_values = field_values.squeeze(-1)

            sym_field_values[:, i, ...] = field_values

        # Average over symmetry operations
        sym_field = np.mean(sym_field_values, axis=1)

        # Reshape symmetrized field back to original shape
        sym_field = sym_field.reshape(field_shape)

        # Validate the symmetrized field
        self.validate_field_properties(sym_field, field_name, original_field)

        self.logger.info(f"Completed real-space symmetrization of {field_name}")
        return sym_field

    def _reciprocal_symmetrize(self, field: np.ndarray, field_name: str) -> np.ndarray:
        """
        Symmetrization in reciprocal space, similar to VASP's approach.
        """
        self.logger.info(f"Starting reciprocal-space symmetrization of {field_name}")

        # Store original field for validation
        original_field = field.copy()

        # Transform to reciprocal space
        field_reciprocal = self._to_reciprocal_space(field)

        # Get reciprocal lattice vectors
        recip_vecs = self._get_reciprocal_vectors()

        # Get grid dimensions
        nx, ny, nz = field.shape[:3]

        # Create reciprocal space grid
        kx = np.fft.fftfreq(nx) * nx  # Scaled to match grid points
        ky = np.fft.fftfreq(ny) * ny
        kz = np.fft.fftfreq(nz) * nz

        # Create meshgrid of k-points
        kgrid = np.array(np.meshgrid(kx, ky, kz, indexing="ij"))

        # Initialize symmetrized field in reciprocal space
        sym_field_reciprocal = np.zeros_like(field_reciprocal)
        weights = np.zeros_like(field_reciprocal, dtype=float)

        # Apply symmetry operations in reciprocal space
        for i, (rot, trans) in enumerate(zip(self.rotations, self.translations)):
            if (i + 1) % 10 == 0 or i == len(self.rotations) - 1:
                self.logger.info(
                    f"Processing symmetry operation {i+1}/{len(self.rotations)}"
                )

            # Rotate k-points
            rot_kgrid = np.einsum("ij,jpqr->ipqr", rot, kgrid)

            # Find corresponding indices in the FFT grid
            indices = np.round(rot_kgrid).astype(int)
            # Apply periodic boundary conditions
            indices = indices % np.array([nx, ny, nz])[:, None, None, None]

            # Compute phase factors from translations
            phase = np.exp(
                -2j * np.pi * np.sum(trans[:, None, None, None] * kgrid, axis=0)
            )

            # Accumulate symmetrized components
            for ix in range(nx):
                for iy in range(ny):
                    for iz in range(nz):
                        idx = (
                            indices[0, ix, iy, iz],
                            indices[1, ix, iy, iz],
                            indices[2, ix, iy, iz],
                        )
                        sym_field_reciprocal[ix, iy, iz] += (
                            field_reciprocal[idx] * phase[ix, iy, iz]
                        )
                        weights[ix, iy, iz] += 1.0

        # Average by weights
        mask = weights > 0
        sym_field_reciprocal[mask] /= weights[mask]

        # Transform back to real space
        sym_field = self._to_real_space(sym_field_reciprocal)

        # Ensure result is real
        if not np.allclose(sym_field.imag, 0, atol=1e-10):
            self.logger.warning("Symmetrized field has non-zero imaginary components")
        sym_field = sym_field.real

        # Validate the symmetrized field
        self.validate_field_properties(sym_field, field_name, original_field)

        self.logger.info(f"Completed reciprocal-space symmetrization of {field_name}")
        return sym_field

    def validate_field_properties(
        self,
        field: np.ndarray,
        field_name: str,
        original_field: Optional[np.ndarray] = None,
    ) -> None:
        """
        Validate physical properties of a field.
        """
        # Check for NaN or infinite values
        if np.any(~np.isfinite(field)):
            raise ValueError(f"{field_name} contains NaN or infinite values")

        # Add unit-aware validation
        if field_name == "density":
            if np.any(field < 0):
                self.logger.error(f"Negative values found in {field_name}")
            self.logger.info(
                f"{field_name} range: [{field.min():.6e}, {field.max():.6e}] e/Å³"
            )
        elif field_name == "kinetic energy density":
            if np.any(field < 0):
                self.logger.error(f"Negative values found in {field_name}")
            self.logger.info(
                f"{field_name} range: [{field.min():.6e}, {field.max():.6e}] eV/Å³"
            )

        # Check total integral conservation
        if original_field is not None:
            volume = np.abs(np.linalg.det(self.atoms.cell))
            nx, ny, nz = field.shape[:3]
            dV = volume / (nx * ny * nz)

            total_orig = np.sum(original_field) * dV
            total_new = np.sum(field) * dV

            relative_diff = (
                abs(total_orig - total_new) / abs(total_orig)
                if abs(total_orig) > 1e-10
                else 0.0
            )

            if relative_diff > 1e-6:
                self.logger.warning(
                    f"Total {field_name} not conserved after symmetrization. "
                    f"Relative difference: {relative_diff:.2e}"
                )
            else:
                self.logger.info(
                    f"Total {field_name} conserved after symmetrization. "
                    f"Relative difference: {relative_diff:.2e}"
                )

    def validate_symmetry(self, field: np.ndarray, label: str) -> None:
        """
        Check if field obeys crystal symmetry, with spatial analysis relative to atomic positions.
        """
        self.logger.info(f"Validating symmetry of {label}")

        # Get field shape and dimensions
        field_shape = field.shape
        spatial_dims = field_shape[:3]
        nx, ny, nz = spatial_dims

        # Create fractional grid coordinates
        grid_points = np.indices(spatial_dims).reshape(3, -1).T / np.array([nx, ny, nz])

        # Sample subset of points for validation
        num_points = 1000
        indices = np.random.choice(len(grid_points), size=num_points, replace=False)
        sampled_points = grid_points[indices]

        # For scalar or vector fields, reshape as needed
        field_reshaped = field.reshape(spatial_dims + (-1,))

        # Grid for interpolation
        grid = (np.arange(nx), np.arange(ny), np.arange(nz))

        # Track violations by region
        violations = {
            "core": [],  # Within 1Å of nuclei
            "bonding": [],  # 1-2Å from nuclei
            "interstitial": [],  # >2Å from nuclei
        }
        max_violation = 0.0

        # For each sampled point, check symmetry
        for idx, point in enumerate(sampled_points):
            # Calculate distance to nearest atom
            dist_to_atom = self._compute_distance_to_atoms(
                point * [nx, ny, nz], (nx, ny, nz)
            )

            # Check symmetry violation
            field_values = []
            for rot, trans in zip(self.rotations, self.translations):
                transformed_point = (rot @ point + trans) % 1.0
                transformed_indices = transformed_point * np.array([nx, ny, nz])
                transformed_indices %= np.array([nx, ny, nz])

                values = []
                for comp in range(field_reshaped.shape[-1]):
                    value = interpn(
                        grid,
                        field_reshaped[..., comp],
                        transformed_indices[np.newaxis, :],
                        method="linear",
                        bounds_error=False,
                        fill_value=0.0,
                    )[0]
                    values.append(value)
                field_values.append(values)

            field_values = np.array(field_values)
            max_diff = np.max(np.ptp(field_values, axis=0))
            max_violation = max(max_violation, max_diff)

    def calculate_elf(self, smooth_sigma: Optional[float] = None) -> np.ndarray:
        """
        Calculate ELF from Wannier functions.
        Working directly in the units of the XSF files (Angstroms).
        """
        self.logger.info("Starting ELF calculation")

        # Calculate volume element
        volume = np.abs(np.linalg.det(self.atoms.cell))
        nx, ny, nz = self.wannier_data[0].shape
        dV = volume / (nx * ny * nz)

        self.logger.info(f"Cell volume: {volume:.6e} Å³")
        self.logger.info(f"Volume element: {dV:.6e} Å³")

        # Initialize arrays (everything in XSF units - Angstroms)
        density = np.zeros_like(self.wannier_data[0])  # e/Å³
        tau = np.zeros_like(density)  # eV/Å³
        grad_density = np.zeros((*density.shape, 3))  # e/Å⁴

        # Process each Wannier function
        for i, wf in enumerate(self.wannier_data):
            self.logger.info(
                f"Processing Wannier function {i+1}/{len(self.wannier_data)}"
            )

            if smooth_sigma is not None:
                wf = gaussian_filter(wf, smooth_sigma)

            # No normalization - Wannier functions should already be normalized

            # Accumulate density (e/Å³)
            density += wf**2

            # Calculate gradients (Å^-4)
            grad_wf = self.compute_gradient(wf)

            # Accumulate kinetic energy density (eV/Å³)
            # Using same prefactor as VASP for consistency
            tau += np.sum(grad_wf**2, axis=-1)

            # Accumulate density gradient (e/Å⁴)
            grad_density += 2 * wf[..., np.newaxis] * grad_wf

        # Double density for non-spin-polarized system
        density *= 2.0

        self.logger.info("Symmetrizing fields...")
        # Symmetrize fields
        density = self.symmetrize_field(density, "density")
        tau = self.symmetrize_field(tau, "kinetic energy density")
        grad_density = self.symmetrize_field(grad_density, "density gradient")

        # Validate symmetry with spatial analysis
        self.validate_symmetry(density, "density")
        self.validate_symmetry(tau, "kinetic energy density")
        self.validate_symmetry(grad_density, "density gradient")

        # Check density values near atomic positions
        """
        self.logger.info("Density values near atomic positions:")
        for i, pos in enumerate(self.atomic_info["grid_positions"]):
            symbol = self.atomic_info["symbols"][i]
            pos_indices = tuple(pos.astype(int) % density.shape)
            density_value = density[pos_indices]
            self.logger.info(
                f"  {symbol} at {tuple(pos_indices)}: {density_value:.6e} e/Å³"
            )
        
        # Integrate density as sanity check
        total_electrons = np.sum(density * dV)
        self.logger.info(f"Total integrated density: {total_electrons:.6f} e")
        """
        self.logger.info("Computing ELF...")
        # Apply density threshold to avoid numerical issues
        density_threshold = 1e-6  # e/Å³
        mask = density > density_threshold

        # Calculate uniform electron gas kinetic energy density
        # Following VASP's approach with same prefactors
        D_h = np.zeros_like(density)
        D_h[mask] = density[mask] ** (5.0 / 3.0)

        # Calculate Pauli kinetic energy term
        grad_density_norm = np.sum(grad_density**2, axis=-1)
        tau_w = np.zeros_like(density)
        tau_w[mask] = grad_density_norm[mask] / (8.0 * density[mask])

        # Calculate D = τ - τ_w
        D = np.maximum(tau - tau_w, 0.0)

        # Initialize ELF array (starting from 0.0, not 0.5)
        elf = np.zeros_like(density)

        # Compute dimensionless χ = D/D_h
        chi = np.zeros_like(density)
        chi[mask] = D[mask] / D_h[mask]

        # Compute ELF
        elf[mask] = 1.0 / (1.0 + chi[mask] ** 2)

        # Print detailed debug information by regions
        self.logger.info("Field ranges by atomic environment:")
        for region in ["core", "bonding", "interstitial"]:
            grid_points = np.indices(density.shape).reshape(3, -1).T
            distances = np.array(
                [
                    self._compute_distance_to_atoms(point, density.shape)
                    for point in grid_points
                ]
            )

            if region == "core":
                mask_region = distances < 1.0
            elif region == "bonding":
                mask_region = (distances >= 1.0) & (distances < 2.0)
            else:  # interstitial
                mask_region = distances >= 2.0

            region_indices = grid_points[mask_region]
            if len(region_indices) > 0:
                self.logger.info(f"\n{region.capitalize()} region statistics:")
                density_region = density[
                    region_indices[:, 0], region_indices[:, 1], region_indices[:, 2]
                ]
                tau_region = tau[
                    region_indices[:, 0], region_indices[:, 1], region_indices[:, 2]
                ]
                elf_region = elf[
                    region_indices[:, 0], region_indices[:, 1], region_indices[:, 2]
                ]

                self.logger.info(
                    f"  Density: [{density_region.min():.6e}, {density_region.max():.6e}] e/Å³"
                )
                self.logger.info(
                    f"  Kinetic energy density: [{tau_region.min():.6e}, {tau_region.max():.6e}] eV/Å³"
                )
                self.logger.info(
                    f"  ELF: [{elf_region.min():.6f}, {elf_region.max():.6f}]"
                )

        self.logger.info("ELF calculation completed")

        # Write fields for visualization
        self.write_field_xsf("density.xsf", density)
        self.write_field_xsf("tau.xsf", tau)
        self.write_field_xsf("tau_w.xsf", tau_w)
        self.write_field_xsf("D_h.xsf", D_h)
        self.write_field_xsf("ELF.xsf", elf)

        return elf

    def write_field_xsf(self, filename: str, field: np.ndarray) -> None:
        """Write a field to an XSF file for visualization."""
        self.logger.info(f"Writing field to file: {filename}")
        write(
            filename,
            self.atoms,
            format="xsf",
            data=field,
            origin=self.origin,
            span_vectors=self.span_vectors,
        )

    def write_elf_xsf(self, filename: Path, elf: np.ndarray) -> None:
        """Write ELF to XSF file."""
        self.logger.info(f"Writing ELF to file: {filename}")
        write(
            filename,
            self.atoms,
            format="xsf",
            data=elf,
            origin=self.origin,
            span_vectors=self.span_vectors,
        )


def main():
    """Main function to handle command line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate ELF from Wannier function XSF files"
    )
    parser.add_argument(
        "input_pattern",
        type=str,
        help='Pattern for input Wannier XSF files (e.g., "wannier*.xsf")',
    )
    parser.add_argument(
        "--output", type=str, default="wan_elf.xsf", help="Output XSF file name"
    )
    parser.add_argument(
        "--gradient-method",
        type=str,
        choices=[m.value for m in GradientMethod],
        default=GradientMethod.FINITE_DIFF.value,
        help="Method for gradient calculation",
    )
    parser.add_argument(
        "--symmetrization-method",
        type=str,
        choices=[m.value for m in SymmetrizationMethod],
        default=SymmetrizationMethod.REAL.value,
        help="Method for symmetrization (real or reciprocal space)",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=None,
        help="Sigma for Gaussian smoothing (optional)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Write logs to file instead of stdout",
    )

    args = parser.parse_args()

    # Setup logging
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_level = logging.DEBUG if args.debug else logging.INFO

    if args.log_file:
        logging.basicConfig(
            level=log_level, format=log_format, filename=args.log_file, filemode="w"
        )
        # Also log to console
        console = logging.StreamHandler()
        console.setLevel(log_level)
        console.setFormatter(logging.Formatter(log_format))
        logging.getLogger("").addHandler(console)
    else:
        logging.basicConfig(level=log_level, format=log_format)

    # Get input files
    input_files = sorted(Path().glob(args.input_pattern))
    if not input_files:
        raise ValueError(f"No files found matching pattern: {args.input_pattern}")

    logging.info(f"Found {len(input_files)} Wannier function files")

    try:
        # Initialize calculator
        calculator = WannierELFCalculator(
            input_files=input_files,
            gradient_method=GradientMethod(args.gradient_method),
            symmetrization_method=SymmetrizationMethod(args.symmetrization_method),
        )

        # Calculate ELF
        elf = calculator.calculate_elf(smooth_sigma=args.smooth_sigma)

        # Write output
        calculator.write_elf_xsf(Path(args.output), elf)

        logging.info("ELF calculation completed successfully")

    except Exception as e:
        logging.error(f"Error during ELF calculation: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
