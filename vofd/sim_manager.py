import os
import numpy as np
from .vo_core import v1, v2, v3, v1_param_eval, v2_param_eval, v3_param_eval
from numba import jit
from functools import partial
from .maxima import max_min
import concurrent.futures


@jit(nopython=True)
def v1_alg(vo_system,q,y0,h):
	return v1(vo_system,q,y0,h)

@jit(nopython=True)
def v2_alg(vo_system,q,y0,h):
	return v2(vo_system,q,y0,h)

@jit(nopython=True)
def v3_alg(vo_system,q,y0,h):
	return v3(vo_system,q,y0,h)

@jit(nopython=True)
def v1_bifurcation(vo_system,q,y0,h,var_):
	y  = v1_param_eval(vo_system,q,y0,h,var_)
	xm = y[0,:] 
	bif_data = max_min(xm,var_)
	return bif_data

@jit(nopython=True)
def v2_bifurcation(vo_system,q,y0,h,var_):
	y  = v2_param_eval(vo_system,q,y0,h,var_)
	xm = y[0,:] 
	bif_data = max_min(xm,var_)
	return bif_data    

@jit(nopython=True)
def v3_bifurcation(vo_system,q,y0,h,var_):
	y  = v3_param_eval(vo_system,q,y0,h,var_)
	xm = y[0,:] 
	bif_data = max_min(xm,var_)
	return bif_data    

def bifurcation_process(vo_alg,vo_system,q,y0,h,delta,L_inf,L_sup):
	it     = round( abs ( (L_sup-L_inf)/delta ) ) 
	var    = [ L_inf + k*delta for k in range(it+1) ]
	xmax   = []
	svar   = []	
	#----------------------------
	partial_process_bifurcation = partial(vo_alg,vo_system,q,y0,h)
	#----------------------------
	max_workers = os.cpu_count() or 1  
	with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
		data = [executor.submit(partial_process_bifurcation,var_) for var_ in var]
	for axis in data:
		saxis = axis.result()
		svar.extend( saxis[0] )
		xmax.extend( saxis[1] )
	return svar,xmax