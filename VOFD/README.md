VOFD Python Package
======================

Overview
--------
The **VOFD Python package** provides tools for simulating and solving **Variable-Order Fractional Derivatives (VOFD)** in dynamical systems. This package is designed for researchers and practitioners to analyze and visualize systems modeled with variable-order derivatives, such as chaotic systems, hereditary processes, and control strategies.

The package supports three main types of VO derivatives:
- **V1: Caputo Variable-Order Derivative**.
- **V2,V3: Variable-Order Derivatives with convolution**.


Overview
--------
VOFD provides high-performance numerical routines accelerated with the **Numba JIT** compiler. The package includes:

- Caputo variable-order derivatives (V1)
- Convolution-based VO derivatives (V2 and V3)
- Simulation of nonlinear and chaotic systems with time-varying fractional order
- Parallel computation of bifurcation diagrams
- Plotting utilities for time-series, phase portraits, and bifurcation structures

The use of **Numba** significantly reduces computation time in derivative evaluation and large-scale bifurcation analysis, enabling efficient exploration of variable-order fractional models.



This package is part of an ongoing research project on variable-order fractional modeling and its applications in nonlinear system analysis. The numerical schemes implemented are based on the finite-difference approximations described in:
B. Parsa Moghaddam and J. A. Tenreiro Machado,
“Extended Algorithms for Approximating Variable Order Fractional Derivatives with Applications,”
Journal of Scientific Computing, 2017.
DOI: 10.1007/s10915-016-0343-1.

Authors

Dr. Daniel Clemente-López
Department of Electronics, Instituto Nacional de Astrofísica, Óptica y Electrónica (INAOE), Mexico

Dr. Jesús M. Muñoz-Pacheco (Corresponding Author)
Faculty of Electronics Sciences, Benemérita Universidad Autónoma de Puebla (BUAP), Mexico

Dr. José de Jesús Rangel-Magdaleno
Department of Electronics, Instituto Nacional de Astrofísica, Óptica y Electrónica (INAOE), Mexico


# Repository Structure

```
VOFD/
│
├── vofd/                  # The Python package
│   ├── __init__.py
│   ├── maxima.py  
│   ├── plot_data.py
│   ├── vo_core.py
│   │__ sim_manager.py
│
├── examples/              # Example scripts
│   ├── simulation_setup.py
│   ├── bifurcation_setup.py
│
├── LICENSE
├── README.md
├── pyproject.toml


# Installation Options

You can install VOFD in two ways:
(1) via PyPI or (2) directly from the source repository.

---

## **Option 1 — Install from PyPI (recommended)**

```bash
pip install vofd
```

Recommended workflow:

1. Create a working directory:

   ```bash
   mkdir VOFD_analysis
   cd VOFD_analysis
   ```

2. (Optional) activate a Python virtual environment:

   **Linux/macOS**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   **Windows**

   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install the package:

   ```bash
   pip install VOFD
   ```

4. Download the example scripts from the GitHub repository (`examples/`) and run:

   ```bash
   python simulation_setup.py
   python bifurcation_setup.py
   ```

---

## **Option 2 — Clone the Repository and Install Locally**

```bash
git clone https://github.com/<your-user>/VOFD.git
cd VOFD
```

Install in editable mode:

```bash
pip install -e .
```

This provides access to:

* Full source code (`vofd/`)
* Example scripts (`examples/`)
* Tests (`tests/`)
* Development configuration

---

Usage
-----
### Simulation Setup (simulation_setup.py)
This script demonstrates how to set up and run a simulation for a --Variable-Order Chaotic Chen System--

1. Setup the system: The script defines the Chen system, a commonly used chaotic system, with initial conditions and a time span.
   
2. Run the simulation: It computes the trajectory of the system using a VO derivative.

3. Output: The script saves the results of the simulation (`y1`, `y2`, `y3`) into `.txt` files and generates a plot of `y1` vs. `y2`. `y1` vs. `y3` and `y2` vs. `y3`.

To run the simulation, simply execute:
```bash
python simulation_setup.py
```

### Bifurcation Analysis Setup (bifurcation_setup.py)
This script is used to perform --bifurcation analysis-- of the --Variable-Order Chaotic Chen System--. 

1. Setup the system: Similar to the simulation script, but this one focuses on analyzing how the system behavior changes as the bifurcation parameter (in this case the `a` parameter), varies.

2. Bifurcation Process: The script calculates bifurcation points for `svar` and `xmax` and stores them in `.txt` files.

3. Output: The bifurcation analysis results are visualized in a plot.

To run the bifurcation analysis, execute:
```bash
python bifurcation_setup.py
```

Key Functions
-------------
**chen_system**: Defines the equations of the Chen system (used in both scripts).
**vo_algorithm**: Variable-order derivative algorithm used for numerical integration.
- In simulation_setup.py: you may use **v1_alg**, **v2_alg**, or **v3_alg**.
- In bifurcation_setup.py: you may use **v1_bifurcation**, **v2_bifurcation**, or **v3_bifurcation**.
**save_time_series**: Saves each state variable (y1, y2, y3) to .txt files (used in simulation_setup.py).
**save_pairwise_plots**: Generates and saves pairwise phase-plane plots, e.g., y1–y2, y1–y3 (used in simulation_setup.py).
**save_bifurcation_data**: Stores the bifurcation values (e.g., parameter sweep and maxima) into .txt files (used in bifurcation_setup.py).
**create_figure**: Creates and saves time-series and phase-portrait plots (used in simulation_setup.py).
**create_bif_figure**: Creates and saves the bifurcation diagram from the computed sweep data (used in bifurcation_setup.py).


# Results Directory

```
results/
```
Plots, time-series, and bifurcation data are stored here.
The folder is generated automatically if it does not exist.

License
-------
The VOFD package is released under the **MIT License**.


