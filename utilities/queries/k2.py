'''for querying k2 with mast'''
from mast import Query as Query_

class Query(Query_):

	def __init__(self):

		self.data_set = "k2/data_search"
		self.mission = 'k2'

		self.base_url = os.path.join(self.archive_url, self.data_set)
		self.search_url = os.path.join(self.base_url, self.search_str)
		self.field_url = self.field_url + 'mission=%s' % self.mission

		self.fields = []

def test(): 
 
	k2 = QueryK2() 
	k2.print_fields() 
	tab = k2.query_MAST(sci_ra='>10.34,<11.34', sci_dec='>-24.45,<-23.45') 
	k2.query_MAST(sci_ra='4.34,14', sci_dec=-24.45) 
	k2.query_MAST(sci_ra=14.34, sci_ec=-24.45) 
	k2.query_MAST(sc_ra=14.34, sci_ec=-24.45) 
	keys = ('K2 ID','Dataset Name','KEP Mag','Object type','Target Type') 
	print keys 
	for i in xrange(100): 
		print [tab[key][i] for key in keys] 
 
def play(): 
 
	k2 = QueryK2() 
	table = k2.query_MAST(sci_campaign=8, kp='>16,>19', imag='>19') 
	keys = ('K2 ID','Dataset Name','KEP Mag','Object type','R Mag') 
	print keys 
	for i in xrange(100): 
		print [table[key][i] for key in keys] 
 
 
if __name__ == '__main__': 
 
	play() 
	#test() 


