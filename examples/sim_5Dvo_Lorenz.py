import numpy as np
from numba import jit
from vofd import v1_alg, save_pairwise_plots, save_time_series

# -------------------------------------------------------
@jit(nopython=True)
def Lorenz_System_5D(y):
    x, y1, z, u, w = y

    a  = 10.0
    b  = 8.0 / 3.0
    c  = 28.0
    h  = -2.0
    k1 = 0.19
    k2 = 12.2

    Dx = a * (y1 - x) + u
    Dy = c * x - x * z + w
    Dz = -b * z + x * y1
    Du = -h * u - x * z
    Dw = -k1 * x - k2 * y1

    return np.array([Dx, Dy, Dz, Du, Dw])
# -------------------------------------------------------

vo_system    = Lorenz_System_5D
vo_algorithm = v1_alg

h     = 0.005
t_sim = 50.0
Stime = np.arange(0, t_sim + h, h)

q = 0.9 + 0.1 * np.sin(np.pi * Stime)

y0 = np.array([[2.0], [3.5], [4.0], [5.0], [7.5]])

y = vo_algorithm(vo_system, q, y0, h)

system_name = f"{vo_system.__name__}_{vo_algorithm.__name__}"
save_pairwise_plots(y, system_name)
save_time_series(y, system_name)
