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
	for i in xrange(5): #Timeouts are common, just need to retry
		try:
			#Set memmap and cache to False if you get an error about too many open files
			hdulist = fits.open(filename, memmap = True, cache = True) 
			return hdulist
		except urllib2.URLError:
			print "Timeout Error, Retrying: Attempt %i/5" % (i+1)
			if not (i+1) % 5:
				print "LC Download Failed, Will Continue Without %s" % str(EPIC)
	return None

def download_target_pixels(EPIC, campaign):

	base_url = 'https://archive.stsci.edu/missions/k2/target_pixel_files/'
	path = 'c%i/%i00000/%02i000/' % (campaign, EPIC / 100000, (EPIC % 100000)/1000)
	filename = 'ktwo%i-c%02i_lpd-targ.fits.gz' % (EPIC, campaign)
	url = base_url + path + filename
	#print url # print if verbose
	hdu = fits.open(url, memmap=True, cache=True)
	return hdu	

def test():

	return download_target_pixels(229228945, 8)
	#return downloadVJ(220213351, 8)

if __name__ == '__main__':

	print test().info()

