
def sff(EPIC, campaign):
	'''retrieve the url of a Vanderburg Johnson Self-Flat Fielded light curve'''
	EPIC = str(EPIC)
	base_url = "http://archive.stsci.edu/missions/hlsp/k2sff"
	path = "c%02i/%s00000/%s" % (campaign, EPIC[:4], EPIC[4:])
	base_name = "hlsp_k2sff_k2_lightcurve_%s-c%02i_kepler_v1_llc.fits" % (EPIC, campaign)
	filename = '/'.join((base_url, path, base_name))
	return filename

def everest(EPIC, campaign):
	'''retrieve the url of an EVEREST processing lighyt curve'''
	EPIC = str(EPIC)
	base_url = "https://archive.stsci.edu/hlsps/everest/v2"
	path = "c%02i/%s00000/%s" % (campaign, EPIC[:4], EPIC[4:])
	base_name = "hlsp_everest_k2_llc_%s-c%02i_kepler_v2.0_lc.fits" % (EPIC, campaign)
	filename = '/'.join((base_url, path, base_name))
	return filename

def sap(EPIC, campaign):
	'''retrieve the url corresponding to a Simple Aperture Photometry light curve'''
	EPIC = str(EPIC)
	base_url = "https://archive.stsci.edu/missions/k2/lightcurves"
	path = "c%i/%s00000/%s000" % (campaign, EPIC[:4], EPIC[4:6])
	base_name = "ktwo%s-c%02i_llc.fits" % (EPIC, campaign)
	filename = '/'.join((base_url, path, base_name))
	return filename


def tpf(EPIC, campaign):
	'''retrieve the url of a Target Pixel File'''
	base_url = 'https://archive.stsci.edu/missions/k2/target_pixel_files/'
	path = 'c%i/%i00000/%02i000/' % (campaign, EPIC / 100000, (EPIC % 100000)/1000)
	filename = 'ktwo%i-c%02i_lpd-targ.fits.gz' % (EPIC, campaign)
	url = base_url + path + filename
	return url


