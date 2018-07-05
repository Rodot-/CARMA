'''
pixellc.py

functions for generating custom representations of K2 "Pixel Light Curves"
from PixMapGenerators to be sent to functions for plotting or further analysis
'''

from itertools import izip, imap, ifilter
import numpy as np


def get_pixel_lc(gen, percentiles, flux_range=None, stat_funcs=None):
	'''return a numpy array of statistics of binned pixels at every cadence

	gen: A PixMapGenerator instance holding data for a CCD
	percentiles: Monotonically increasing list of percentile bin edges (0-100)
	flux_range: 2-tuple of flux ranges to cut the pixel values at
	stat_funcs: list of functions to pass binned data to

	returns: N x M x (2+L) array
		N: Number of epochs
		M: Number of percentile *bins*; len(percentiles) - 1
		L: Number of statistics to compute
			(L+2): First 2 arrays are the bin edges
	'''

	if flux_range is not None:
		lower, higher = sorted(flux_range)

	if stat_funcs is None:
		stat_funcs = (np.median,)

	ccd = gen.ccd
	N = gen.N
	M = len(percentiles)-1
	L = len(stat_funcs)

	lc = np.empty((N, M, L+2))
	lc[::] = np.nan

	l_bin_edges = np.empty((gen.N, len(percentiles)))

	def _get_epochs():

		if flux_range is not None:
			return _get_filtered_epochs()

		return _get_full_epochs()

	def _get_full_epochs():

		for i, g in enumerate(imap(gen.get_unordered, xrange(gen.N))):

				yield g

	def _get_filtered_epochs():

		for i, g in enumerate(imap(gen.get_unordered, xrange(gen.N))):

			mask = (g > lower) & (g < higher)
			yield g[mask]

	epochs = list(_get_epochs())

	lengths = np.array(map(len, epochs))
	unique_lengths = np.unique(lengths[lengths!=0])

	for length in unique_lengths:

		l_index = np.where(lengths == length)[0]
		l_uniform_array = np.asarray([epochs[i] for i in l_index]) 

		l_bin_edges[l_index] = np.percentile(l_uniform_array, percentiles, axis=1).T
		l_bin_edges[l_index,-1] += 1e-10

		lc[l_index,:,0] = l_bin_edges[l_index,:-1]
		lc[l_index,:,1] = l_bin_edges[l_index,1:]	

		for lc_epoch, y, bin_edges in izip(lc[l_index], l_uniform_array, l_bin_edges[l_index]):		

			indices = np.digitize(y, bin_edges)-1 # digitize offsets the index

			bin_counts = np.bincount(indices) # try to arrange arrays so that we can compute stats in parallel
			same_counts = np.unique(bin_counts[bin_counts!=0])

			for count in same_counts:

				index = np.where(bin_counts == count)[0]
				uniform_array = np.asarray([y[i==indices] for i in index])

				lc_epoch[index, 2:] = np.array([stat_func(uniform_array, axis=1) for stat_func in stat_funcs]).T
	
	return lc

def get_pixel_lc_old(gen, percentiles, mag_range=None):
	'''percentiles is a flat array of percentiles ranging from 0 to 100'''
	N = gen.N
	higher, lower = None, None
	if mag_range is None:
		higher, lower = np.inf, -np.inf
		mag_range = (lower, higher)
	else:
		higher, lower = magToFlux(np.array(sorted(mag_range)))
	if np.inf in mag_range:
		lower = -np.inf
	
	ccd = gen.ccd
	M = len(percentiles)-1
	
	lc = np.empty((N,M,5)) # min, max, var, median, mean
	lc[::] = np.nan
		
	funcs = (np.var, np.median, np.mean)
	N = len(gen.containers.containers[0].pixels)
	
	for i,g in enumerate((gen.get_unordered(i) for i in xrange(N))):
		m = (g > lower) & (g < higher)
		if m.any():
			p = np.percentile(g[m], percentiles)
			lc[i,:,:2] = zip(p[:-1], p[1:])
			
			# New
			for j, (low, high) in enumerate(lc[i,:,:2]):
				if j+1 == M:
					high += 1
				cut = g[(g >= low) & (g < high)]
				if len(cut):
					lc[i,j,2] = np.var(cut)
					lc[i,j,3] = np.median(cut)
					#lc[i,j,4] = np.dot(cut,np.ones(len(cut)))/len(cut)
					lc[i,j,4] = np.mean(cut)
					#lc[i,j,5] = len(cut)

	return lc


import time
class Timeit:


	def __init__(self, name=None):

		self.name = (' '+name) if name is not None else ''

	def __enter__(self):

		self.T0 = time.clock()

	def __exit__(self, *args):

		print "Time to run{}:".format(self.name), time.clock() - self.T0

def test():
	from .. import containers
	from ..ccd import CCD
	import pdb	

	import time
	print "Making CCD"
	ccd = CCD(campaign=8, module=6, channel=2, field='FLUX')
	print "  ", ccd
	print "Making Container"
	cont = containers.PixelMapContainer.from_hdf5('K2PixelMap.hdf5', ccd)
	print "  ", cont
	print "Making Generator"
	gen = containers.PixMapGenerator(cont)
	print "  ", gen

	percentiles = np.linspace(0,100,1000)

	def len_func(x, axis=0):
		return [len(i) for i in x]

	stat_funcs = (np.var, np.median, np.mean)
	#stat_funcs = None
	print "Making lc"
	T0 = time.clock()
	lc = get_pixel_lc(gen, percentiles, stat_funcs=stat_funcs)
	#print "  ", lc
	print "   Made lc in {} seconds".format(time.clock()-T0)
	T0 = time.clock()
	lc2 = get_pixel_lc_old(gen, percentiles)
	print "   Made lc2 in {} seconds".format(time.clock()-T0)

	print "Max Diff:", np.nanmax(np.abs(lc-lc2))
	print "Sum Diff:", np.nansum(np.abs(lc-lc2))	

	pdb.set_trace()

if __name__ == '__main__':

	#benchmark()
	test()
