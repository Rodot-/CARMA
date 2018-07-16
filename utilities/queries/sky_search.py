'''
sky_search.py

Search rectangular areas of the sky using MAST
Also includes a search for multiple EPIC IDs
'''

from k2 import Query
from .. import k2fov
from concurrent.futures import ThreadPoolExecutor

k2query = Query()
def rect_search(rect, split='ra'):
	'''search a rectangular area of the sky defined by the rectangle
	defined by (top, right, bottom, left)
	'''
	
	top, right, bottom, left = rect
	ra_bounds = "{}..{}".format(left, right)
	dec_bounds = "{}..{}".format(bottom, top)
	query_result = list(k2query.query_MAST(sci_ra=ra_bounds, sci_dec=dec_bounds, coordformat='dec'))
	TPE = ThreadPoolExecutor(2)
	
	if len(query_result) == 2002: # too many objects returned
		
		if split == 'ra': # left right splitting
			dx = (right-left)/2
			dy = 0
			split = 'dec'
			
		elif split == 'dec': # top bottom splitting
			dx = 0
			dy = (top-bottom)/2
			split = 'ra'
		
		f1 = TPE.submit(rect_search, (top-dy, right-dx, bottom, left), split=split)
		f2 = TPE.submit(rect_search, (top, right, bottom+dy, left+dx), split=split)
		return f1.result() + f2.result()[2:]
		
	#print "Queried ra={}, dec={}".format(ra_bounds, dec_bounds)
	#print "  Found {} objects".format(len(query_result))
	return query_result
		
def epic_search(EPICs):
	'''Search MAST by EPIC Ids'''

	TPE = ThreadPoolExecutor(64)
	chunks = (EPICs[i:i+500] for i in xrange(0, len(EPICs), 500))
	strings = (','.join(map(str, chunk)) for chunk in chunks)
	futures = [TPE.submit(k2query.query_MAST, ktc_k2_id=s, coordformat='dec') for s in strings]
	header = None
	query = []
	for future in futures:
		res = list(future.result())
		if header is None: header = res[:2]
		query.extend(res[2:])
	return header + query

def test():
	
	fov = k2fov.Field(8)
	query_result = rect_search(fov.bounding_box)
	print "Found {} objects!".format(len(query_result))
	table = k2query.to_table(iter(query_result))
	EPICs = map(int, table['K2 ID'])
	print "Running Epic Search"
	q2 = epic_search(EPICs)
	print "Found {} objects!".format(len(q2))
	
