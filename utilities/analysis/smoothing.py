'''
smoothing.py

A small collection of functions for smoothing datasets
'''
import numpy as np
from functools import partial

def running_func(func):
	'''applies a function for smoothing over a window'''
	def _wrapper(x, N, **func_kwargs):
		'''The smoothing function
		x: input data array
		N: Smoothing window
		'''
		n, m = N/2, N%2
		return np.array([func(x[i-n if i > n else i:i+m+n], **func_kwargs) for i in xrange(len(x))])

	return _wrapper

running_mean = running_func(np.nanmean)
running_median = running_func(np.nanmedian)
running_std = running_func(np.nanstd)


