import numpy as np
from numba import jit
from vofd import v1_alg, create_xy_figure, save_time_series
#-------------------------------------------------------
@jit(nopython=True)
def Ricatti(y):
	Dq_y = -y*y + 2*y + 1
	return Dq_y 
#-------------------------------------------------------

# ======================================================
# 2. Solver selection 
# ======================================================

vo_system 	 = Ricatti
vo_algorithm = v1_alg


# ======================================================
# 3. Time discretization
# ======================================================

h     = 0.01 # step size
t_sim = 5  # simulation time
Stime = np.arange(0, t_sim + h, h)        #Time span array


# ======================================================
# 4. Variable order definition & Initial conditions 
# ======================================================

q     = 0.6 - 0.1*np.sin(np.pi*Stime)     #Variable order
y0    = np.array([ [0.0] ]) #Initial Conditions


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
