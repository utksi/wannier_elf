# ELF with wannier orbitals

Calculates ELF (and all the stuff in between wannier functions and ELF) with wannier orbitals.

## Equations

Given a set of wannier functions \{$w_n(\mathbf{r})$\}, we can get:

1. **Electron Density**: 
   ```math
   \rho(\mathbf{r}) = 2 \sum_n |w_n(\mathbf{r})|^2
   ```

2. **Kinetic Energy Density**:
   ```math
   \tau(\mathbf{r}) = \frac{1}{2} \sum_n |\nabla w_n(\mathbf{r})|^2
   ```

3. **Pauli Kinetic Energy Density**:
   ```math
   \tau_p(\mathbf{r}) = \tau(\mathbf{r}) - \frac{1}{8} \frac{|\nabla\rho(\mathbf{r})|^2}{\rho(\mathbf{r})}
   ```

4. **Homogeneous Electron Gas Kinetic Energy Density**:
   ```math
   \tau_h(\mathbf{r}) = \frac{3}{10}(3\pi^2)^{2/3} \rho(\mathbf{r})^{5/3}
   ```

5. **Electron Localization Function (ELF)**:
   ```math
   \text{ELF}(\mathbf{r}) = \frac{1}{1 + \left(\frac{\tau_p(\mathbf{r})}{\tau_h(\mathbf{r})}\right)^2}
   ```
   ELF is a dimensionless quantity ranging from 0 to 1, indicating electron localization.
   This form in a localized basis is identical to LOL (Localized orbital locator).

## Order of Execution
 
This is for personal use so there is no CLI.

1. **Calculate Electron Density**: 
   - **Script**: `e_density.py`
   - **Description**: Reads Wannier function `.xsf` files, computes the electron density, and saves it to `.npy` and `.xsf` files.
   - **Output**: `electron_density.npy`, `electron_density.xsf`

2. **Calculate Kinetic Energy Density**:
   - **Script**: `ke_density.py`
   - **Description**: Computes the kinetic energy density from Wannier functions and saves it to `.npy` and `.xsf` files.
   - **Output**: `kinetic_energy_density.npy`, `kinetic_energy_density.xsf`

3. **Calculate Pauli Kinetic Energy Density**:
   - **Script**: `pauli_ke_density.py`
   - **Description**: Uses the electron density and kinetic energy density to compute the Pauli kinetic energy density.
   - **Output**: `pauli_ke_density.npy`, `pauli_ke_density.xsf`

4. **Calculate Homogeneous Electron Gas Kinetic Energy Density**:
   - **Script**: `heg_ke_density.py`
   - **Description**: Computes the kinetic energy density for a homogeneous electron gas from the electron density.
   - **Output**: `heg_ke_density.npy`, `heg_ke_density.xsf`

5. **Calculate Electron Localization Function (ELF)**:
   - **Script**: `elf_calculator.py`
   - **Description**: Computes the ELF using the Pauli and HEG kinetic energy densities.
   - **Output**: `elf.npy`, `elf.xsf`

6. **Visualization**:
   - **Script**: `plot_slice.py`
   - **Description**: Plots slices of ELF along different planes (XY, YZ, ZX) at depth = 0.5.
   - **Output**: Writes .png files.

6. **Interactive Visualization**:
   - **Script**: `interactive_slice.py`
   - **Description**: Provides interactive visualization of ELF slices using Plotly, allowing exploration along different planes (XY, YZ, ZX).
   - **Output**: Interactive plot displayed in a web browser.

## Requirements

- **Python Libraries**:
  - ASE: For reading and writing XSF files.
  - NumPy: For numerical calculations.
  - Matplotlib: For static plotting (optional).
  - Plotly: For interactive visualization (optional).

## Installation

Ensure all required Python libraries are installed:

```bash
python -m venv ~/.venv/wannier_elf
source ~/.venv/wannier_elf/bin/activate
python -m pip install ase numpy matplotlib plotly