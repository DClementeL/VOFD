import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path

def save_output():
	results_dir = Path.cwd() / "results"
	results_dir.mkdir(exist_ok=True)


def symmetric_bounds(data, step, center=0.0):
	max_dev = np.max(np.abs(data - center))
	lim = np.ceil(max_dev / step) * step
	return center - lim, center + lim

def x_smart_bounds(data, step, neg_tol=0.05):
	dmin = np.min(data)
	dmax = np.max(data)

	span = dmax - dmin
	if span == 0:
		return dmin - step, dmax + step

	
	if dmin >= 0 or abs(dmin) / span < neg_tol:
		upper = np.ceil(dmax / step) * step
		return dmin, upper

	# Truly bipolar → symmetric
	max_dev = max(abs(dmin), abs(dmax))
	lim = np.ceil(max_dev / step) * step
	return -lim, lim


def y_smart_bounds(data, step, neg_tol=0.05):
	dmin = np.min(data)
	dmax = np.max(data)

	span = dmax - dmin
	if span == 0:
		return dmin - step, dmax + step

	
	if dmin >= 0 or abs(dmin) / span < neg_tol:
		upper = np.ceil(dmax / step) * step
		return 0, upper

	# Truly bipolar → symmetric
	max_dev = max(abs(dmin), abs(dmax))
	lim = np.ceil(max_dev / step) * step
	return -lim, lim



def auto_tick_step(vmin, vmax, target_ticks=8):
	span = abs(vmax - vmin)
	if span == 0:
		return 1.0

	raw = span / target_ticks
	exp = np.floor(np.log10(raw))
	base = raw / 10**exp

	if base <= 1:
		step = 1
	elif base <= 2:
		step = 2
	elif base <= 5:
		step = 5
	else:
		step = 10

	return step * 10**exp




def create_xy_figure(xy, spec=None):
	save_output()
	# ---------------- default configuration -------------
	cfg = {
		"x_label": "y1",
		"y_label": "y2",
		"figure_name": "phase_plot",
		"plot_type": "line",      # "line" or "scatter"
		"color": "blue",
		"thickness": 1.4,
		"grid": True
	}

	Dont_touch = {
		"neg_tol": 0.05,
		"target_ticks": 8,
	}
	if spec:
		cfg.update(spec)

	# ---------------- data validation -------------------
	a = np.asarray(xy, dtype=float)
	if a.ndim != 2 or a.shape[0] != 2:
		raise ValueError(
			f"xy must have shape (2, N). Got {a.shape}"
		)

	x_data = a[0, :]
	y_data = a[1, :]

	# ---------------- bounds ----------------------------
	x_step_tmp = auto_tick_step(np.min(x_data), np.max(x_data),
							Dont_touch["target_ticks"])
	# X axis: always symmetric (chaotic convention)
	if np.min(x_data) < 0:
		x0, x1 = symmetric_bounds(x_data, x_step_tmp)
	else:
		x0, x1 = x_smart_bounds(x_data, x_step_tmp, Dont_touch["neg_tol"])
	# Y axis: smart decision
	y_step_tmp = auto_tick_step(np.min(y_data), np.max(y_data),
								Dont_touch["target_ticks"])
	y0, y1 = y_smart_bounds(y_data, y_step_tmp, Dont_touch["neg_tol"])

	# Recompute final steps after snapping
	x_step = auto_tick_step(x0, x1, Dont_touch["target_ticks"])
	y_step = auto_tick_step(y0, y1, Dont_touch["target_ticks"])

	# ---------------- plot ------------------------------
	plt.figure(figsize=(8, 6))

	if cfg["plot_type"] == "line":
		plt.plot(x_data, y_data,
				 color=cfg["color"],
				 linewidth=cfg["thickness"])
	elif cfg["plot_type"] == "scatter":
		plt.scatter(x_data, y_data,
					color=cfg["color"],
					s=0.05)
	else:
		raise ValueError(f"Unsupported plot_type: {cfg['plot_type']}")

	plt.xlabel(cfg["x_label"], fontsize=26)
	plt.ylabel(cfg["y_label"], fontsize=26)

	plt.xlim(x0, x1)
	plt.ylim(y0, y1)

	# ---------------- ticks -----------------------------
	ax = plt.gca()
	ax.xaxis.set_major_locator(MultipleLocator(x_step))
	ax.yaxis.set_major_locator(MultipleLocator(y_step))

	ax.tick_params(axis="both", which="major",
				   labelsize=18, length=6, width=1.2)

	# ---------------- grid ------------------------------
	if cfg["grid"]:
		plt.grid(True, linewidth=0.8, alpha=0.6)

	# ---------------- save ------------------------------
	plt.savefig(f"results/{cfg['figure_name']}.png",
				dpi=420, bbox_inches="tight")
	plt.show()
	plt.close()



def save_pairwise_plots(data_matrix,system_name):
	save_output()

	n = data_matrix.shape[0]
	var_names = [f"y{i+1}" for i in range(n)]

	for i in range(n):
		for j in range(i+1, n):
			cfg = {
				"x_label": var_names[i],
				"y_label": var_names[j],
				"figure_name": f"{system_name}_plot_{var_names[i]}_{var_names[j]}",
				"plot_type": "line",      # "line" or "scatter"
				"color": "blue",
				"thickness": 1.4,
				"grid": True
			}			
			data = np.vstack((data_matrix[i], data_matrix[j]))
			create_xy_figure(data,cfg)

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