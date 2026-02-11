import numpy as np
from numba import jit
from vofd import (
	v1_bifurcation,
	v2_bifurcation,
	v3_bifurcation,
	bifurcation_process,
	create_xy_figure,
	save_bifurcation_data
)

# ======================================================
# 1. Variable-order system definition (NUMBA SAFE)
# ======================================================

@jit(nopython=True)
def Lorenz_System_5D(y, bif_param):
	x, y1, z, u, w = y

	a  = 10.0
	b  = 8.0 / 3.0
	c  = 28.0
	h  = -2.0
	k1 = 0.19
	k2 = bif_param

	Dx = a * (y1 - x) + u
	Dy = c * x - x * z + w
	Dz = -b * z + x * y1
	Du = -h * u - x * z
	Dw = -k1 * x - k2 * y1

	return np.array([Dx, Dy, Dz, Du, Dw])
# -------------------------------------------------------

# ======================================================
# 2. Solver selection 
# ======================================================

vo_system   = Lorenz_System_5D
vo_algorithm = v3_bifurcation


# ======================================================
# 3. Time discretization
# ======================================================

h     = 0.005
t_sim = 30.0
Stime = np.arange(0, t_sim + h, h)


# ======================================================
# 4. Variable order definition & Initial conditions 
# ======================================================

q = 0.9 + 0.1 * np.sin(np.pi * Stime)
y0 = np.array([[2.0], [3.5], [4.0], [5.0], [7.5]])


# ======================================================
# 5. Bifurcation parameter sweep
# ======================================================

L_inf = 0
L_sup = 70.0
delta = 0.1


# ======================================================
# 6. Solver invocation
# ======================================================

if __name__ == "__main__":

	X = bifurcation_process(
		vo_algorithm,
		vo_system,
		q,
		y0,
		h,
		delta,
		L_inf,
		L_sup
	)

	# ==================================================
	# 7. Post-processing
	# ==================================================

	system_name = f"{vo_system.__name__}_{vo_algorithm.__name__}"

	plot_param = {
		"x_label": "k2",
		"y_label": "y1",
		"figure_name": system_name,
		"plot_type": "scatter",      # "line" or "scatter"
		"color": "blue",
		"grid": False
	}
	create_xy_figure(X, plot_param)

	save_bifurcation_data(X, system_name)