from downloader import fits_downloader
from urls import *

def test():

	urls = [[get(229228945,8) for get in (sff, everest, tpf, sap)] for _ in xrange(4)]
	for item in fits_downloader(urls, nthreads=20, ordered=False):
		for hdu in item:
			print hdu


if __name__ == '__main__':
	test()
