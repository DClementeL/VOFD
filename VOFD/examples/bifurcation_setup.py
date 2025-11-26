import numpy as np
from numba import jit
from vofd import v1_bifurcation, bifurcation_process, create_bif_figure, save_bifurcation_data

@jit(nopython=True)
def Chen_System(y,var):
	y1,y2,y3 = y
	a = var   
	b = 3  
	c = 28
	Dqy1 = a*(y2-y1)
	Dqy2 = (c-a)*y1 - y1*y3 + c*y2
	Dqy3 = y1*y2 - b*y3
	return np.array([Dqy1,Dqy2,Dqy3])

#-------------------------------------------------------
vo_system 	 = Chen_System
vo_algorithm = v1_bifurcation
h     = 0.005 # step size
t_sim = 30.0  # simulation time
Stime = np.arange(0, t_sim + h, h)        #Time span array
q     = 0.9 + 0.1*np.sin(np.pi*Stime)     #Variable order
y0    = np.array([ [1.0], [0.0], [1.0] ]) #Initial Conditions
#-------------------------------------------------------
L_inf = 40
L_sup = 60
delta = 0.1
#-------------------------------------------------------
if __name__ == '__main__':
    X = bifurcation_process(vo_algorithm,vo_system,q,y0,h,delta,L_inf,L_sup)
    #-------------------------------------------------------

    # save results
    system_name = f"{vo_system.__name__}_{vo_algorithm.__name__}"
    create_bif_figure(data=(X[0],X[1]), axis_names=('a','x1_max'), figure_name=f"{system_name}")
    save_bifurcation_data(X,system_name)
