'''
K2FOV.py - Tools for working with k2 fields

Author: Jack O'Brien
Date: 10/19/2017

Execution: python 2.7
Requires: numpy, matplotlib, JacksTools

Usage:
	Can plot a field from a campaign
	Can determine if an object lies within a field
	Tools for placing objects within a k2 campaign field
'''


from JacksTools import jio
import numpy as np
from matplotlib.pyplot import subplots, show

from multiprocessing import Process, Pipe, cpu_count, Queue
from multiprocessing.queues import SimpleQueue, JoinableQueue

FOOTPRINT_FILE="../data/k2-footprint.csv"

class Field:
	''' A single K2 campaign feild
		Relevant methods for end user:
			plot_field
			test_point
	This thing sort of works as a state machine underneath
	'''
	data = jio.load(FOOTPRINT_FILE, headed=True, delimiter=',')
	# This array of tuples iterates over adjacent edges of a square
	grouping = zip(np.arange(4), (np.arange(4)+1)%4)

	def __init__(self, campaign, cache_data=True):

		self.campaign = campaign
		self.cache = cache_data
		self.footprint = self._get_campaign(campaign)
		self.length = len(self.footprint)
		self.ra = np.array([self.footprint['ra'+i] for i in '0123'])
		self.dec = np.array([self.footprint['dec'+i] for i in '0123'])
		self.bounding_box = self.__get_bounding_box()
		self.bbox_set = tuple(map(self.__get_bounding_box, xrange(len(self))))
		mask = self.footprint['module'] == 13 # Module 13 is the center module
		self.center = {'ra':np.mean(self.ra[:,mask]), 'dec':np.mean(self.dec[:,mask])}
	
	def plot_field(self, ax, label=False): # TODO: update this to use self.ra and self.dec
		''' plot the footprint on an axis '''
		fp = self.footprint # For shorter typing 
		sqr = np.arange(5)%4 # indeces of box corners
		for idx, ccd in enumerate(self): # Iterate over the CCDs
			# Grab the box corners, complete the shape by repeating first term
			ax.plot(self.ra[sqr, idx], self.dec[sqr, idx], color='k', lw=1)
		if label: # You can set the label to True or a string
			if type(label) is bool:
				label="Campaign %i" % self.campaign
			ax.text(self.center['ra'], self.center['dec'], str(self.campaign), fontsize = 25)
			ax.scatter([self.center['ra']], [self.center['dec']])

	def test_point(self, ra, dec):
		''' test if a point is inside a field, return the channel number '''

		top, right, bottom, left = self.bounding_box
		if not (left < ra < right) or not (bottom < dec < top): #Check Full Field bbox
			return 0
		for idx, ccd in enumerate(self): # Iterate over the channels
			top, right, bottom, left = self.bbox_set[idx]
			if left < ra < right and bottom < dec < top: #Check channel bbox
				test = self.__test_channel(idx, ra, dec)
				t0 = test.next() # Get the first truth value
				for t in test:
					if t is not t0: # All we care about is all elements are uniform
						break
				else:
					return ccd['channel']
		return 0

	@classmethod
	def _get_campaign(self, campaign):
		''' get the footprint of a campaign '''
		fp = self.data[self.data['campaign'] == campaign]
		# Place RA on scale from -180 to 180 in degrees
		if any(ra > 180 for i in '0123' for ra in fp['ra'+i]):
			for i in '0123': fp['ra'+i][::] -= 360
		return fp

	def __test_channel(self, idx, ra, dec):
		''' Test if a point at ra and dec are within the bbox of a channel 
			This works by taking vectors describing the channel edges and
			taking the cross product with a vector defined by the point relative
			to the first point in the edge vector.  If the cross product of these
			vectors is the same sign for all edges, then the point must be contained
			within the box.  This method is a generator which yeilds a boolean
			describing the sign of the cross product (True for positive, 
			False for negative)
		'''
		ccd_ra, ccd_dec = self.ra[:,idx], self.dec[:,idx]
		for i,j in self.grouping:
			# Compute the edge vector
			dra, ddec = ccd_ra[j] - ccd_ra[i], ccd_dec[j] - ccd_dec[i]
			# Compute the point to vertex vector
			dpra, dpdec = ra - ccd_ra[i], dec - ccd_dec[i]
			# Compute the cross product
			pos = dra*dpdec - ddec*dpra
			yield pos > 0

	def __get_bounding_box(self, idx=None):
		''' get a box that bounds the edge of the field (t,r,b,l)'''
		if idx is None: 
			ra, dec = self.ra, self.dec
		else:
			ra, dec = self.ra[:,idx], self.dec[:,idx]
		top = dec.max()
		right = ra.max()
		bottom = dec.min()
		left = ra.min()
		return top, right, bottom, left	

	def __len__(self):
		''' just get the number of elements in the footprint '''
		return self.length

	def __iter__(self):
		''' iterate over elements of the footprint list '''
		for ccd in self.footprint:
			yield ccd

def test2():
	
	import sys
	
	args = sys.argv[1:]
	doShow = False
	campaign = 8
	for i,arg in enumerate(args):
		if arg in ('-s','--show'):
			doShow = True
		if arg[:2] == '-c':
			if arg[2:]:
				campaign = int(arg[2])
			else:
				campaign = int(args[i+1])
	sdss = jio.fits.open("../../Data/mastercat/GTR-ADM-QSO-master-sweeps-Feb5-2016.zspec.fits")[1].data
	RA = sdss['RA']
	RA[RA > 180] -= 360
	DEC = sdss['DEC']
	field = Field(campaign)
	print("Testing Master Catalog on Campaign %i" % campaign)
	ra, dec = zip(*(coord for coord in zip(RA, DEC) if field.test_point(*coord)))
	print("%i Points Found in Campaign %i" % (len(ra), campaign))
	if campaign == 8:
		assert len(ra) == 12805
	if doShow:
		fig, ax = subplots(1,1)
		ax.scatter(ra,dec)
		field.plot_field(ax, True)
		show()

if __name__ == '__main__':

	test()



