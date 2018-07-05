'''
plotting.py

A small collection of convenient plotting functions
'''
from smoothing import *

def axes_prop_setter(local_vars, *args, **kwargs):

	locals().update(local_vars)
	for key, value in kwargs.items():
		if hasattr(value, '__iter__'):
			value = iter(value)
		elif type(value) is str:
			if value.startswith('@'): # for custom expressions
				value = eval(value[1:])
		else:
			value = iter([value])
		eval('ax.{}(*value)'.format(key))
	for arg in args:
		eval('ax.{}()'.format(arg))

def plot_nice_intervals(ax, x, y, N=30, label=None, color=None, scatter=False):

	if color is None:
		color = next(ax._get_lines.propr_cycler)['color']

	rm = running_median(y, N)

	if scatter:
		rs = running_std(y, N)
		ax.plot(x, y, ls=' ', marker='.', ms=1, color=color)
		ax.fill_between(x, rm+rs, rm-rs, color=color, alpha=0.1, linestyle='None')

	ax.plot(x, rm, ls='-', color=color, lw=2, label=label)		




