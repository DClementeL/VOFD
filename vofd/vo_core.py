import numpy as np
import math as mt
from numba import jit

@jit(nopython=True)
def v1(vo_system,q,y0,h):
	n    = len(q)
	dim  = len(y0)
	y    = np.zeros((dim,n))
	y[:,0] = y0[:,0]
	sum_ = np.zeros(dim)
	for k in range(n-1):						
		v1D = vo_system(y[:,k])
		hq   = pow(h,q[k])*mt.gamma(2 - q[k])
		sum_.fill(0)
		for j in range(1,k+1):
			psi   = pow( j+1, 1 - q[k] ) - pow(j, 1 - q[k] ) 
			Delta_y = np.zeros(dim)	
			for z in range(dim):
				Delta_y[z] = y[z,k-j+1] - y[z,k-j]
				sum_[z]   += psi*Delta_y[z]
		y[:,k+1] = hq*v1D - sum_ + y[:,k]
	return y

@jit(nopython=True)
def v2(vo_system,q,y0,h):
	n    = len(q)
	dim  = len(y0)
	y    = np.zeros((dim,n))
	y[:,0] = y0[:,0]
	#-----
	varphi = np.zeros(n)
	hq     = np.zeros(n)
	sum_ = np.zeros(dim) 
	for j in range (0,n-1):
		varphi[j] = pow( j+1, 1 - q[j]) - pow(j, 1 - q[j])
		hq[j] = pow(h,-q[j])/( mt.gamma( 2 - q[j] ) )
	hq0 = pow(h,q[0])*mt.gamma( 2 - q[0] )		
	for k in range(n-1):						
		v2D = vo_system(y[:,k])
		sum_.fill(0)  
		for j in range(1,k+1):
			Delta_y = np.zeros(dim)	
			for z in range(dim):
				Delta_y[z] = y[z,k-j+1] - y[z,k-j]
				sum_[z] += hq[j]*varphi[j]*Delta_y[z]		
		y[:,k+1] = hq0*(v2D - sum_) + y[:,k]  
	return y

@jit(nopython=True)
def v3(vo_system,q,x0,h):
	n      = len(q)
	dim    = len(x0)
	y      = np.zeros((dim,n))
	y[:,0] = x0[:,0]

	hq0 = pow(h,q[0])*(1 - q[0])*mt.gamma(1 - q[0])

	sum_ = np.zeros(dim)
	Delta_y  = np.zeros(dim)

	for k in range(n-1):		
		v3D  = vo_system(y[:,k])				
		sum_.fill(0) 
		for  j in range(1,k+1):			
			Lambda_s = pow(j+1, 1 - q[k-j]) - pow(j, 1 - q[k-j]) 
			hq_s     = pow(h,-q[k-j])/mt.gamma(2 - q[k-j])  							
			Delta_y.fill(0)
			for z in range(dim):
				Delta_y[z] = y[z,k-j+1] - y[z,k-j]
				sum_[z] += hq_s*Lambda_s*Delta_y[z]
		y[:,k+1] = hq0*(v3D - sum_) + y[:,k]
	return y


@jit(nopython=True)
def v1_param_eval(vo_system,q,y0,h,var_):
	n    = len(q)
	dim  = len(y0)
	y    = np.zeros((dim,n))
	y[:,0] = y0[:,0]
	sum_ = np.zeros(dim)
	for k in range(n-1):						
		v1D = vo_system(y[:,k],var_)
		hq   = pow(h,q[k])*mt.gamma(2 - q[k])
		sum_.fill(0)
		for j in range(1,k+1):
			psi   = pow( j+1, 1 - q[k] ) - pow(j, 1 - q[k] ) 
			Delta_y = np.zeros(dim)	
			for z in range(dim):
				Delta_y[z] = y[z,k-j+1] - y[z,k-j]
				sum_[z]   += psi*Delta_y[z]
		y[:,k+1] = hq*v1D - sum_ + y[:,k]
	return y

@jit(nopython=True)
def v2_param_eval(vo_system,q,y0,h,var_):
	n    = len(q)
	dim  = len(y0)
	y    = np.zeros((dim,n))
	y[:,0] = y0[:,0]
	#-----
	varphi = np.zeros(n)
	hq     = np.zeros(n)
	sum_ = np.zeros(dim) 
	for j in range (0,n-1):
		varphi[j] = pow( j+1, 1 - q[j]) - pow(j, 1 - q[j])
		hq[j] = pow(h,-q[j])/( mt.gamma( 2 - q[j] ) )
	hq0 = pow(h,q[0])*mt.gamma( 2 - q[0] )		
	for k in range(n-1):						
		v2D = vo_system(y[:,k],var_)
		sum_.fill(0)  
		for j in range(1,k+1):
			Delta_y = np.zeros(dim)	
			for z in range(dim):
				Delta_y[z] = y[z,k-j+1] - y[z,k-j]
				sum_[z] += hq[j]*varphi[j]*Delta_y[z]		
		y[:,k+1] = hq0*(v2D - sum_) + y[:,k]  
	return y

@jit(nopython=True)
def v3_param_eval(vo_system,q,x0,h,var_):
	n      = len(q)
	dim    = len(x0)
	y      = np.zeros((dim,n))
	y[:,0] = x0[:,0]

	hq0 = pow(h,q[0])*(1 - q[0])*mt.gamma(1 - q[0])

	sum_ = np.zeros(dim)
	Delta_y  = np.zeros(dim)

	for k in range(n-1):		
		v3D  = vo_system(y[:,k],var_)				
		sum_.fill(0) 
		for  j in range(1,k+1):			
			Lambda_s = pow(j+1, 1 - q[k-j]) - pow(j, 1 - q[k-j]) 
			hq_s     = pow(h,-q[k-j])/mt.gamma(2 - q[k-j])  							
			Delta_y.fill(0)
			for z in range(dim):
				Delta_y[z] = y[z,k-j+1] - y[z,k-j]
				sum_[z] += hq_s*Lambda_s*Delta_y[z]
		y[:,k+1] = hq0*(v3D - sum_) + y[:,k]
	return y