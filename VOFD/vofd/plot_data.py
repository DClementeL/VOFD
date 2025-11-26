import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def save_output():
	results_dir = Path.cwd() / "results"
	results_dir.mkdir(exist_ok=True)


def create_figure(data, axis_names, figure_name):
	save_output()
	# Extract the X & Y data
	x_data, y_data = data
	x_label, y_label = axis_names
	
	# Default parameters
	plot_type = 'line'
	color     = 'blue'
	thickness = 1
	#plt.figure(figsize=(8, 6))
	
	if plot_type == 'line':
		plt.plot(x_data, y_data, color=color, linewidth=thickness)
	elif plot_type == 'scatter':
		plt.scatter(x_data, y_data, color=color)
	elif plot_type == 'bar':
		plt.bar(x_data, y_data, color=color)
	else:
		raise ValueError("Invalid plot type. Choose 'line', 'scatter', or 'bar'.")
	
	
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.grid(True)
	plt.show(block=False)
	plt.savefig(f'results/{figure_name}.png', bbox_inches='tight', dpi=420)    
	input('Press Enter to close')
	plt.close()



def create_bif_figure(data, axis_names, figure_name):
	save_output()
	# Extract the X & Y data
	x_data, y_data = data
	x_label, y_label = axis_names

	plt.scatter(x_data,y_data, marker='.', s=1, color = 'black',linewidth=0.1)    
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.grid(True)
	plt.show(block=False)
	plt.savefig(f'results/{figure_name}.png', bbox_inches='tight', dpi=420)    
	input('Press Enter to close')
	plt.close()



def save_pairwise_plots(data_matrix,system_name):
	save_output()

	n = data_matrix.shape[0]
	var_names = [f"y{i+1}" for i in range(n)]

	for i in range(n):
		for j in range(i+1, n):
			create_figure(
				data=(data_matrix[i], data_matrix[j]),
				axis_names=(var_names[i], var_names[j]),
				figure_name=f"{system_name}_plot_{var_names[i]}_{var_names[j]}")
			

def save_time_series(y,system_name):
	save_output()
	for idx in range(y.shape[0]):
		np.savetxt(
			f"results/{system_name}_y{idx+1}_TimeSeries.txt",
			y[idx, :],
			delimiter=","
		)
		  

def save_bifurcation_data(y,system_name):
		save_output()
		np.savetxt(
			f"results/{system_name}_bif_parameter.txt",
			y[0],
			delimiter=","
		)
			
		np.savetxt(
			f"results/{system_name}_bif_state.txt",
			y[1],
			delimiter=","
		)          