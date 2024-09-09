# ELF with wannier orbitals

Calculates ELF (and all the stuff in between wannier functions and ELF) with wannier orbitals.

## Equations

Given a set of wannier functions $\{\{w_n(\mathbf{r})\}\}$, we can get [See Nature, 371(1994)683-686](https://www.nature.com/articles/371683a0) for a non spin-polarized system:

$$D = -2*\sum{w_n^* \nabla^2w_n} + \frac{1}{2}\nabla^2 n  -  \frac{1}{4*n}\left(\nabla n\right)^2 .$$

$$D_{H} = 0.6*\left(6*\pi^2\right)^{2/3}*n^{5/3} .$$

$$ELF\ (r) = \frac{1}{1 + \bigg(\frac{D}{D_H}\bigg)^2} .$$

## How to:
 
This is for personal use so there is no CLI.

1. **Run**: 
   - **Script**: `elf.py`
   - **Description**: Reads Wannier function `.xsf` files, computes the ELF, and saves it to a `.xsf` file. Need options file `options.dat`. 
   - `options.dat` allows settings for KE density symmetrization (not recommended for calculating ELF), interpolation (recommended), normalization (not needed, here for a sanity check), use_fft (use fft to calculate D -> highly recommended).
   - **Output**: `elf.xsf`

2. **Visualization**:
   - **Script**: `plot_slice.py`
   - **Description**: Plots slices of ELF along different planes (XY, YZ, ZX) at depth = 0.5.
   - **Output**: Writes .png files.

3. **VESTA/pyMOL/VMD/Ovito**:
   - Visualize the 'elf.xsf' file in anything that supports it.

4. **Interactive slice**:
   - **Script**: `interactive_slice.py`
   - **Description**: Provides interactive visualization of ELF slices using Plotly, allowing exploration along different planes (XY, YZ, ZX).
   - **Output**: Interactive plot displayed in a web browser.

## Requirements

- **Python Libraries**:
  - ASE: For reading and writing XSF files.
  - NumPy: For numerical calculations.
  - Spglib: For symmetrizing charge density.
  - tqdm: Progressbar
  - Matplotlib: For static plotting (optional).
  - Plotly: For interactive visualization (optional).

## Installation

Ensure all required Python libraries are installed:

```bash
python -m venv ~/.venv/wannier_elf
source ~/.venv/wannier_elf/bin/activate
python -m pip install ase numpy matplotlib plotly spglib tqdm
