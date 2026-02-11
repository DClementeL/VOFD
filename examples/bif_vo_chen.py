import numpy as np
from numba import jit
from vofd import (
    v1_bifurcation,
    bifurcation_process,
    create_xy_figure,
    save_bifurcation_data
)

# ======================================================
# 1. Variable-order system definition (NUMBA SAFE)
# ======================================================

@jit(nopython=True)
def Chen_System(y, bif_param):
    y1, y2, y3 = y
    a = bif_param
    b = 3
    c = 28

    return np.array([
        a * (y2 - y1),
        (c - a) * y1 - y1 * y3 + c * y2,
        y1 * y2 - b * y3
    ])


# ======================================================
# 2. Solver selection 
# ======================================================

vo_system   = Chen_System
vo_algorithm = v1_bifurcation


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
y0 = np.array([[1.0], [0.0], [1.0]])


# ======================================================
# 5. Bifurcation parameter sweep
# ======================================================

L_inf = 40.0
L_sup = 60.0
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
        "x_label": "a",
        "y_label": "$y1_{max}$",
        "figure_name": system_name,
        "plot_type": "scatter",      # "line" or "scatter"
        "color": "blue",
        "grid": False
    }
    create_xy_figure(X,plot_param)

    save_bifurcation_data(X, system_name)