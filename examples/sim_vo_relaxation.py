import numpy as np
from numba import jit
from vofd import v1_alg, v2_alg, v3_alg, create_xy_figure, save_time_series
#-------------------------------------------------------
@jit(nopython=True)
def Relaxation(y):
	Dq_y = -y
	return Dq_y 
#-------------------------------------------------------

# ======================================================
# 2. Solver selection 
# ======================================================

vo_system 	 = Relaxation
vo_algorithm = v3_alg


# ======================================================
# 3. Time discretization
# ======================================================

h     = 0.01 # step size
t_sim = 5  # simulation time
Stime = np.arange(0, t_sim + h, h)        #Time span array


# ======================================================
# 4. Variable order definition & Initial conditions 
# ======================================================

q0, q1 = 0.55, 0.95
q = q0 + (q1 - q0)*(Stime / t_sim)

y0    = np.array([ [1.0] ]) #Initial Conditions


# ======================================================
# 6. Solver invocation
# ======================================================

y   = vo_algorithm(vo_system,q,y0,h)
y   = np.vstack((Stime, y))
# ==================================================
# 7. Post-processing
# ==================================================

system_name = f"{vo_system.__name__}_{vo_algorithm.__name__}"

plot_param = {
	"x_label": "Time(s)",
	"y_label": "y",
	"figure_name": system_name,
	"plot_type": "line",      # "line" or "scatter"
	"color": "blue",
	"thickness": 1.4,
	"grid": False
}

create_xy_figure(y,plot_param)
save_time_series(y,system_name)