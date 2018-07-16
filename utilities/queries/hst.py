'''for querying hst with mast'''
from mast import Query as Query_
import os

class Query(Query_):

	def __init__(self):

		self.data_set = "hst"
		self.mission = "hst"

		self.base_url = os.path.join(self.archive_url, self.data_set)
		self.search_url = os.path.join(self.base_url, self.search_str)

		self.field_url = self.field_url + 'mission=%s' % self.mission

		self.fields = []

def test(): 
 
	hst = Query() 
	hst.print_fields() 
	tab = hst.to_table(hst.query_MAST(sci_ra='10.34..11.34', sci_dec='-24.45..-23.45'))
	hst.query_MAST(sci_ra='4.34,14', sci_dec=-24.45) 
	hst.query_MAST(sci_ra=14.34, sci_ec=-24.45) 
	hst.query_MAST(sc_ra=14.34, sci_ec=-24.45) 
	keys = ('Dataset','Proposal ID','Target Name') 
	print tab.keys()
	for i in xrange(len(tab.values()[0])): 
		print [tab[key][i] for key in keys] 
 
if __name__ == '__main__': 
 
	test() 


