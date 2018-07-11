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

	return ax.plot(x, rm, ls='-', color=color, lw=2, label=label)		


def plot_pixel_lc(ax, lc, smooth=False, image=False, differenced=False, **kwargs):
	'''Plot a pixel lc as either a set of light curves or an image
	in which rows represent light curves

	args:
	ax - matplotlib Axes object to plot to
	lc - NxM numpy array of light curves
	smooth - Bool or int of boxcar smoothing window over which to smooth the light curve
	image - Bool, whether to plot individual light curves or make an image plot
	differenced - Bool, whether to difference the rows of the light curve
	cmap - matplotlib color map to apply to the image
	'''

	# create the image from the rows of the light curve
	image_ = lc if not differenced else lc[:, 1:] - lc[:, :-1]
	
	# whiten the light curve
	image_ -= np.nanmedian(image_, axis=0)
	image_ /= np.nanstd(image_, axis=0)

	# optionally smooth the light curve
	if smooth:
		N = 30 if smooth is True else smooth
		image_[::] = running_median(image_, N, axis=0)
	
	if image:
		return ax.imshow(image_.T[::-1], **kwargs)
	
	return [ax.plot(line, **kwargs) for line in image_.T] 


def format_pixel_image(ax, im, n_epochs, cadence=0.2043229):
	'''apply some additional formatting to the pixel image plot
	These basically apply my own formattings like I prefer'''
	
	ax.set_yticks(np.linspace(0, ax.get_ybound()[1]+1, 10))
	ax.set_xticks(np.linspace(0, ax.get_xbound()[1]+1, 10))

	ax.set_yticklabels(np.linspace(100,0,11).astype(int))
	ax.set_xticklabels(map('{:.0f}'.format,np.arange(11)*cadence*n_epochs/11))

	ax.set_xlabel('Time (days)', fontsize=24)
	ax.set_ylabel('Percentile', fontsize=24)

	cbar = ax.figure.colorbar(im, ticks=np.linspace(-1,1,3), pad=0.1)

	ax.tick_params(axis='both', which='both', labelsize=24) 


