import numpy as np
from numba import jit
from vofd import v1_alg, save_pairwise_plots, save_time_series
#-------------------------------------------------------
@jit(nopython=True)
def Chen_System(y,var=None):
	y1,y2,y3 = y
	a = 40
	b = 3
	c = 28
	Dqy1 = a*(y2-y1)
	Dqy2 = (c-a)*y1 - y1*y3 + c*y2
	Dqy3 = y1*y2 - b*y3
	return np.array([Dqy1,Dqy2,Dqy3])
#-------------------------------------------------------

vo_system 	 = Chen_System
vo_algorithm = v1_alg
h     = 0.005 # step size
t_sim = 30.0  # simulation time
y0    = np.array([ [1.0], [0.0], [1.0] ]) #Initial Conditions
Stime = np.arange(0, t_sim + h, h)        #Time span array
q     = 0.9 + 0.1*np.sin(np.pi*Stime)     #Variable order
#-------------------------------------------------------

y   = vo_algorithm(vo_system,q,y0,h)
system_name = f"{vo_system.__name__}_{vo_algorithm.__name__}"
save_pairwise_plots(y,system_name)
save_time_series(y,system_name)

