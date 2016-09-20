import numpy as np
from enum import Enum

import bokeh.plotting as plt
from bokeh.models import Range1d
from bokeh.plotting import ColumnDataSource
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color

default_width = 900
default_height = 450

class InterpolatedColorScheme:
	"""
	Given a list of RGB colors, defines a function that maps the interval $[0, 1]$ to a set of
	colors that interpolates smoothly between the values in the provided list. This is done by
	converting the colors to LAB and constructing a cubic interpolator for each channel.
	"""

	def __init__(self, rgb_colors):
		assert len(rgb_colors) >= 2
		self.lab_colors = [convert_color(c, LabColor) for c in rgb_colors]

		x = np.linspace(0, 1, len(self.lab_colors))
		y_l = np.array([c.lab_l for c in self.lab_colors])
		y_a = np.array([c.lab_a for c in self.lab_colors])
		y_b = np.array([c.lab_b for c in self.lab_colors])

		self.p_l = np.polyfit(x, y_l, 3)
		self.p_a = np.polyfit(x, y_a, 3)
		self.p_b = np.polyfit(x, y_b, 3)

	def __call__(self, t):
		assert t >= 0 and t <= 1
		l, a, b = [np.polyval(p, t) for p in [self.p_l, self.p_a, self.p_b]]

		rgb = convert_color(LabColor(l, a, b), sRGBColor)
		r, g, b = [round(c) for c in rgb.get_upscaled_value_tuple()]
		return 'rgb({},{},{})'.format(r, g, b)


red_color_values = ['#fff7ec', '#fee8c8', '#fdd49e', '#fdbb84', '#fc8d59', '#ef6548', '#d7301f',
	'#b30000', '#7f0000']

blue_color_values = ['#fff7fb', '#ece7f2', '#d0d1e6', '#a6bddb', '#74a9cf', '#3690c0', '#0570b0',
	'#045a8d', '#023858']

blue_to_red_color_values = list(reversed(blue_color_values)) + red_color_values

red = InterpolatedColorScheme([sRGBColor.new_from_rgb_hex(c) for c in red_color_values])

blue_to_red = InterpolatedColorScheme([sRGBColor.new_from_rgb_hex(c) for c in
	blue_to_red_color_values])

def make_line_plot(t, x, title, width=default_width, height=default_height):
	fig = plt.figure(title=title, tools='save', plot_width=width, plot_height=height)
	fig.line(t, x)
	return fig

def _to_decibels(X):
	return 20 * np.log10(X / np.max(X))

def make_spectogram(X, stride_ms, title, width=default_width, height=default_height,
	color_scheme=red):

	assert len(X.shape) == 2
	assert np.all(X > 0)
	window_count, window_length = X.shape[0], X.shape[1]

	scaled_X = _to_decibels(np.ravel(X))
	times = np.repeat(stride_ms / 1000 * np.arange(0, window_count), window_length)
	assert len(times) == len(scaled_X)

	spec_min, spec_max = np.min(scaled_X), np.max(scaled_X)
	a = 1 / (spec_max - spec_min)
	b = -a * spec_min

	bins = np.tile(np.arange(0, window_length), window_count)
	bin_colors = [color_scheme(a * s + b) for s in scaled_X]
	source = ColumnDataSource(data=dict(times=times, bins=bins, colors=bin_colors))

	print(bin_colors[1:10])
	
	fig = plt.figure(title=title, x_range=Range1d(0, times[-1]),
		y_range=Range1d(0, window_length - 1), x_axis_label='Time (seconds)',
		y_axis_label='Frequency Bin', plot_width=width, plot_height=height, tools='save')

	fig.grid.grid_line_color = None
	fig.axis.axis_line_color = None
	fig.axis.major_tick_line_color = None
	fig.rect('times', 'bins', stride_ms / 1000, 1, source=source, color='colors')
	return fig

def make_mel_spectogram(coeffs, stride_ms, title, width=default_width, height=default_height,
	color_scheme=blue_to_red):

	assert len(coeffs.shape) == 2
	window_count, window_length = coeffs.shape[0], coeffs.shape[1]

	coeffs_flat = np.ravel(coeffs)
	times = np.repeat(stride_ms / 1000 * np.arange(0, window_count), window_length)
	assert len(times) == len(coeffs_flat)

	"""
	Let $l$ and $u$ be the minimum and maximum coefficients, respectively, and suppose that
	$u > 0 > l$. We define a linear map $\phi : [l, u] \to [0, 1]$, such that $\phi(0) = 1 / 2$.
	This is done so that a coefficient of zero (representing a lack of "energy" for a particular
	channel) maps to the "neutral" color of the provided color scheme (likely to be white).
	"""
	
	min_coeff, max_coeff = np.min(coeffs_flat), np.max(coeffs_flat)
	index, _ = max(enumerate([abs(min_coeff), abs(max_coeff)]), key=lambda x: x[1])
	m = [min_coeff, max_coeff][index]
	assert m != 0, "All coefficients are zero."

	a = 1 / (2 * m) if m > 0 else -1 / (2 * m)
	b = 1 / 2

	bins = np.tile(np.arange(0, window_length), window_count)
	bin_colors = [color_scheme(a * s + b) for s in coeffs_flat]
	source = ColumnDataSource(data=dict(times=times, bins=bins, colors=bin_colors))
	
	fig = plt.figure(title=title, x_range=Range1d(0, times[-1]),
		y_range=Range1d(0, window_length - 1), x_axis_label='Time (seconds)',
		y_axis_label='Frequency Bin', plot_width=width, plot_height=height, tools='save')

	fig.grid.grid_line_color = None
	fig.axis.axis_line_color = None
	fig.axis.major_tick_line_color = None
	fig.rect('times', 'bins', stride_ms / 1000, 1, source=source, color='colors')
	return fig
