'''
pixellc.py

functions for generating custom representations of K2 "Pixel Light Curves"
from PixMapGenerators to be sent to functions for plotting or further analysis
'''

from itertools import izip
import numpy as np
from multiprocessing.pool import ThreadPool

def _apply_stat_func(apply_stat_func_info):

	y, stat_func = apply_stat_func_info
	return stat_func(y)

def _get_cut_mask(get_cut_info):

	i, indices = get_cut_info
	mask = indices == i
	return mask	

def _get_cuts(get_cuts_info):

	indices, lc, y = get_cuts_info
	for i, lc_p in enumerate(lc):
		mask = indices == i
		#cut_mask_args = [(i, indices) for i in xrange(len(lc))]
		#for mask, lc_p in izip(P4.imap(_get_cut_mask, cut_mask_args), lc):
		if mask.any():
			yield y[mask], lc_p

P = ThreadPool(8) # computing stats
P2 = ThreadPool(8) # computing over epochs
P3 = ThreadPool(8) # computing unordered data
P4 = ThreadPool(8) # compute masks

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

	def _filter_lc_epoch(epoch):

		i, y = epoch

		if flux_range is not None:
			mask = (y > lower) & (y < higher)
		else:
			mask = np.ones(len(y), dtype=np.bool)	

		if mask.any():

			return y[mask], lc[i]

	def _digitize_epoch(epoch):

		y, lc = epoch
		p = np.percentile(y, percentiles)
		p[-1] += 1e-14 # to make sure we include all data
		lc[:,:2] = zip(p[:-1], p[1:])

		indices = np.digitize(y, p)
	
		def _cut_epochs(index):

			mask = index == indices
			if mask.any():

				def _compute_stat(stat_func):

					return stat_func(y[mask])

				lc[index][2:] = P4.map(_compute_stat, stat_funcs)
		
		return (i for i in P2.imap(_cut_epochs, xrange(len(lc))) if i is not None)


	epochs = ((i,gen.get_unordered(i)) for i in xrange(gen.N))
	good_epochs = (i for i in P.imap(_filter_lc_epoch, epochs) if i is not None)
	cut_epochs = P3.imap(_digitize_epoch, good_epochs)	
	for cut_epoch in cut_epochs:
		pass

	return lc


def test():
	from .. import containers
	from ..ccd import CCD
	import pdb	

	import time
	print "Making CCD"
	ccd = CCD(campaign=8, module=6, channel=2, field='FLUX')
	print "  ", ccd
	print "Making Container"
	cont = containers.PixelMapContainer.from_hdf5('K2PixelMap_Camp8_Mod6_Chan2.hdf5', ccd)
	print "  ", cont
	print "Making Generator"
	gen = containers.PixMapGenerator(cont)
	print "  ", gen

	percentiles = np.linspace(0,100,50)
	stat_funcs = (np.median, np.mean, np.std)
	print "Making lc"
	T0 = time.clock()
	lc = get_pixel_lc(gen, percentiles, stat_funcs=stat_funcs)
	#print "  ", lc
	print "   Made lc in {} seconds".format(time.clock()-T0)
	pdb.set_trace()

if __name__ == '__main__':

	test()
