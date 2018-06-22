'''
Downloads LCs from various locations
'''
import sys
import urllib2
import gzip
from itertools import imap, izip
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, Future
from io import BytesIO
from astropy.io import fits
import types

# Downloading backends

def download_helper(url, **kwargs):
	'''returns a raw file that can be opened by astropy fits'''

	try:
		socket = urllib2.urlopen(url)
		raw = socket.read()
	except urllib2.URLError as e:
		print >>sys.stderr, e, url
		print >>sys.stderr, "Reason:",e.reason
		return url, None
	finally:
		try:
			socket.close()
		except NameError:
			pass
	return raw

def downloader(urls, ordered=True, nthreads=20):
	'''download a list of urls'''
	assert nthreads > 0, "nthreads must be greater than 0, but got '{}'".format(nthread)
	assert type(nthreads) is int, "nthreads must be in integer but got '{}'".format(type(nthreads))

	download_func = lambda _urls: imap(download_helper, _urls)

	if nthreads > 1:
		threaded_downloader = threaded(pool=None, results=True, nthreads=nthreads)(download_helper)
		download_func = lambda _urls: threaded_downloader(urls, ordered=ordered)

	return download_func(urls)

def fits_downloader(urls, ordered=True, nthreads=20):
	'''Download a list of urls and return fits hdus.

	Returns a generator [of generators] of fits hdu lists

	Arguments:
		urls -- a string or [nested] iterable of strings representing urls

	Keyword arguments:
		ordered -- output maintains order of urls (default True) 
		nthreads -- number of maximum concurrent download threads (default 20)
	
	Additional notes:
		* Setting `ordered` to `False` will allow the downloader to retrieve
		files much more rapidly at the expense of random ordering of outputs.
		* Setting nthreads too high may result in errors as the targeted server
		may deny requests, though the GIL bottleneck of the fits processing
		should limit this.
		* When nested iterables of urls are submitted, nested generators of 
		generators are returned by the function.

	'''
	assert nthreads > 0, "nthreads must be greater than 0, but got '{}'".format(nthread)
	assert type(nthreads) is int, "nthreads must be in integer but got '{}'".format(type(nthreads))

	def fits_helper(url, **kwargs):
		'''helper method for downloading and opening a single fits file'''
		is_gz = url.endswith('.gz')
		raw = download_helper(url, **kwargs)
		hdu = open_raw_fits(raw, gz=is_gz)
		return hdu		

	download_func = lambda _urls: imap(fits_helper, _urls)

	if nthreads > 1:
		threaded_downloader = threaded(pool=None, results=True, nthreads=nthreads)(fits_helper)
		download_func = lambda _urls: threaded_downloader(urls, ordered=ordered)

	return download_func(urls)

def threaded(pool=None, results=False, nthreads=20):
	'''makes a function over an iterable threaded
	submits all items to the pool ahead of time, so
	we don't have to reinstantiate anything'''

	pool = pool
	results = results
	nthreads=nthreads
	def decorator(func, pool=pool):

		if pool is None:
			pool = ThreadPoolExecutor(nthreads)

		def submitter(iterable, **kwargs):
			futures = []
			for item in iterable:
				if hasattr(item, '__iter__') and (type(item) is not str):
					futures.append(submitter(item, **kwargs))
				else:
					futures.append(pool.submit(func, item, **kwargs))
			return futures

		def retriever(futures, **kwargs):

			ordered = kwargs.setdefault('ordered', True)
			flat = all([type(future) is Future for future in futures])	
			iterator = iter(futures)
			if flat:
				if not ordered: iterator = as_completed(futures)
				for future in iterator:
					if ordered: wait([future])
					yield future.result()
			else:
				for item in iterator:
					if type(item) is Future:
						if ordered: wait([item])
						yield future.result()
					else:
						yield retriever(item, **kwargs)
				
		def threaded_func(iterable, **kwargs):

			futures = submitter(iterable, **kwargs)
			return retriever(futures, **kwargs)

		wrapper = threaded_func if results else submitter 

		wrapper.__doc__ = func.__doc__
		return wrapper

	return decorator

@threaded(pool=None, results=True, nthreads=100)
def concurrent_downloader(url, **kwargs):

	return download_helper(url)
	
def open_raw_fits(raw, gz=False):

	if raw is None:
		return None
	if gz: # the file is gzip compressed
		stream = BytesIO(raw)
		gz = gzip.GzipFile(None, 'rb', 9, stream)
		buff = BytesIO(gz.read())
		gz.close()
	else:
		buff = BytesIO(raw)
	hdu = fits.open(buff, cache=False)
	return hdu

