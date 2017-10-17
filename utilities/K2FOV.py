'''
K2FOV.py

Tools for working with k2 fields

Can plot a field from a campaign
Can determine if an object lies within a field
Tools for placing objects within a k2 campaign field
'''


from JacksTools import jio
import numpy as np
from matplotlib.pyplot import subplots, show

FOOTPRINT_FILE="../data/k2-footprint.csv"

class Field:
	''' A single K2 campaign feild
		Relevant methods for end user:
			plot_field
			test_point
	'''
	data = jio.load(FOOTPRINT_FILE, headed=True, delimiter=',')

	def __init__(self, campaign):

		self.campaign = campaign
		self.footprint = self.get_campaign(campaign)
		self.ra = np.array([self.footprint['ra'+i] for i in '0123'])
		self.dec = np.array([self.footprint['dec'+i] for i in '0123'])
		self.bounding_box = self.get_bounding_box()

	def test_point(self, ra, dec):
		''' test if a point is inside a field, return the channel number '''
		fp = self.footprint
		top, right, bottom, left = self.bounding_box
		idx = np.arange(4)
		grouping = zip(idx, (idx+1)%4)
		contained = np.empty(4, dtype=bool)
		if left < ra < right and bottom < dec < top: #Check Full Field bbox
			for ccd in fp:
				top, right, bottom, left = self.get_bounding_box(ccd)
				if left < ra < right and bottom < dec < top: #Check channel bbox
					idx = (i for i in xrange(4))
					cra, cdec = self.get_coordinates(ccd)
					for i,j in grouping: #Check inside channel
						dra, ddec = cra[j] - cra[i], cdec[j] - cdec[i]
						dpra, dpdec = ra - cra[i], dec - cdec[i]
						pos = dra*dpdec - ddec*dpra
						contained[idx.next()] = pos > 0
					if all(contained) or not any(contained):
						return ccd['channel']
		return 0

	@classmethod
	def get_campaign(self, campaign):
		''' get the footprint of a campaign '''
		fp = self.data[self.data['campaign'] == campaign]
		# Place RA on scale from -180 to 180 in degrees
		if any(ra > 180 for i in '0123' for ra in fp['ra'+i]):
			for i in '0123': fp['ra'+i][::] -= 360
		return fp

	def get_coordinates(self, ccd):
		''' get the coordinates of the channel in array form '''
		ra = np.array([ccd['ra'+i] for i in '0123'])
		dec = np.array([ccd['dec'+i] for i in '0123'])
		return ra, dec

	def plot_field(self, ax, label=False):
		''' plot the footprint on an axis '''
		fp = self.footprint # For shorter typing 
		for ccd in fp: # Iterate over the CCDs
			# Grab the box corners, complete the shape by repeating first term
			ra, dec = zip(*((ccd['ra'+i], ccd['dec'+i]) for i in '01230')) 
			ax.plot(ra, dec, color='k', lw=1)
		if label: # You can set the label to True or a string
			if type(label) is bool:
				label="Campaign %i" % self.campaign
			ax.text(np.mean(fp['ra0']), np.mean(fp['dec0']), str(self.campaign), fontsize = 25)

	def get_bounding_box(self, ccd=None):
		''' get a box that bounds the edge of the field (t,r,b,l)'''
		if ccd is None: 
			top = self.dec.max()
			right = self.ra.max()
			bottom = self.dec.min()
			left = self.ra.min()
		else:
			ra, dec = self.get_coordinates(ccd)
			top = dec.max()
			right = ra.max()
			bottom = dec.min()
			left = ra.min()
		return top, right, bottom, left	

def test():

	fig, ax = subplots(1,1)
	field = Field(0)
	assert field.test_point(94.94, 27.3773) == 72 # should return 72
	assert field.test_point(194.94, -87.3773) == 0 # should return 0 (not in field)
	for c in xrange(0,17):
		Field(c).plot_field(ax, True)
	show()

if __name__ == '__main__':

	test()



