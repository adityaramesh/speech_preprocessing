import numpy as np
import bokeh.plotting as plt
from bokeh.models import Range1d
from bokeh.plotting import ColumnDataSource

default_width = 900
default_height = 450

def make_line_plot(t, x, title, width=default_width, height=default_height):
	fig = plt.figure(title=title, tools='save', plot_width=width, plot_height=height)
	fig.line(t, x)
	return fig

spectogram_colors = ['#f7f4f9', '#e7e1ef', '#d4b9da', '#c994c7', '#df65b0', '#e7298a', '#ce1256',
	'#980043', '#67001f']

def make_spectogram(X, window_stride_ms, title, width=default_width, height=default_height):
	assert len(X.shape) == 2
	window_count, window_length = X.shape[0], X.shape[1]

	log_X = np.log(np.ravel(X))
	times = np.repeat(window_stride_ms / 1000 * np.arange(0, window_count), window_length)
	assert len(times) == len(log_X)
	
	spec_min, spec_max = np.min(log_X), np.max(log_X)
	a = (len(spectogram_colors) - 1) / (spec_max - spec_min)
	b = -a * spec_min

	bins = np.tile(np.arange(0, window_length), window_count)
	bin_colors = [spectogram_colors[int(round(a * s + b))] for s in log_X]
	source = ColumnDataSource(data=dict(times=times, bins=bins, colors=bin_colors))
	
	fig = plt.figure(title=title, x_range=Range1d(0, times[-1]),
		y_range=Range1d(0, window_length - 1), x_axis_label='Time (seconds)',
		y_axis_label='Frequency Bin', plot_width=width, plot_height=height, tools='save')

	fig.grid.grid_line_color = None
	fig.axis.axis_line_color = None
	fig.axis.major_tick_line_color = None
	fig.rect('times', 'bins', window_stride_ms / 1000, 1, source=source, color='colors')
	return fig
