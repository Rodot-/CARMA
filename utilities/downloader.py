'''
Downloads LCs from various locations
'''

import urllib2
from astropy.io import fits

#Vanderberg LCs

def downloadVJ(EPIC, campaign):
	EPIC = str(EPIC)
	base_url = "http://archive.stsci.edu/missions/hlsp/k2sff"
	path = "c%02i/%s00000/%s" % (campaign, EPIC[:4], EPIC[4:])
	base_name = "hlsp_k2sff_k2_lightcurve_%s-c%02i_kepler_v1_llc.fits" % (EPIC, campaign)
	filename = '/'.join((base_url, path, base_name))
	result = None
	for i in xrange(5): #Timeouts are common, just need to retry
		try:
			hdulist = fits.open(filename, memmap = True, cache = True) #Set memmap and cache to False if you get an error about too many open files
			return hdulist
		except urllib2.URLError:
			print "Timeout Error, Retrying: Attempt %i/5" % (i+1)
			print "LC Download Failed, Will Continue Without %s" % str(EPIC)
	return None

def test():

	downloadVJ(220213351, 8)

if __name__ == '__main__':

	test()

