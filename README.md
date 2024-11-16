# ELF with wannier orbitals

Calculates ELF (and all the stuff in between wannier functions and ELF) with wannier orbitals.

## What?

Given a set of wannier functions $\{\{w_n(\mathbf{r})\}\}$, we can get [See Nature, 371(1994)683-686](https://www.nature.com/articles/371683a0) for a non spin-polarized system:

$$D = -2*\sum{w_n^* \nabla^2w_n} + \frac{1}{2}\nabla^2 n  -  \frac{1}{4*n}\left(\nabla n\right)^2 .$$

$$D_{H} = 0.6*\left(6*\pi^2\right)^{2/3}*n^{5/3} .$$

$$ELF\ (r) = \frac{1}{1 + \bigg(\frac{D}{D_H}\bigg)^2} .$$

## How?
 
This is for personal use so there is no CLI.

Run:

```bash
python elf.py "wannier*.xsf" --symmetrization-method=reciprocal
```

for reciprocal space symmetrization of density, density gradients and kinetic energy density
(highly preferred to obtain small errors after symmetrization).

```bash
python elf.py "wannier*.xsf" --symmetrization-method=real
```

for real space symmetrization of density, density gradients and kinetic energy density
(highly preferred to obtain small errors after symmetrization).


## Input

Needs wannier*.xsf files generated via VASP

## Output

ELF.xsf and all scalar fields needed to calculate the ELF.

Can be visualized by VESTA, CrystalMaker, or any other viewer for scalar fields

## Requirements

- **Python Libraries**:
  - ASE: For reading and writing XSF files.
  - NumPy: For numerical calculations.
  - Spglib: For symmetrizing charge density.
  - tqdm: Progressbar

## Installation

Ensure all required Python libraries are installed:

```bash
python -m venv ~/.venv/wannier_elf
source ~/.venv/wannier_elf/bin/activate
python -m pip install ase numpy spglib tqdm
```
