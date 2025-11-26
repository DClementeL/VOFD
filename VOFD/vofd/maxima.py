import numpy as np
from numba import jit

@jit(nopython=True)
def max_min(x,var):
	xmax    = []
	svar    = []
	go = round ( len(x)*0.35 ) 
	if (x[-1] < 1e03)  and (x[-1] > -1e03):
		for i in range( go, len(x)-1 ):
			if x[i - 1] < x[i] and x[i] > x[i + 1]:
				xmax.append( x[i] )
				svar.append( var )
	return svar,xmax
