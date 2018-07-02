'''a series of general containers and iterators for working with K2 data'''
import os
import gc
import numpy as np
import h5py
from copy import copy
from ccd import CCD
from downloads.downloader import fits_downloader
from downloads.urls import tpf
from context import LoadingBar
from . import data
from queries import search
from k2fov import Field

# We'll look at functions for extracting target pixels based on the module

def moduleFilter(ccd):
	K2FOV = Field(ccd.campaign)
	channel = Field.get_channels(ccd.module)[ccd.channel]
	def _filter(item):
		ra, dec = item[1][1:]
		if ra > 180: ra -= 360
		mod, chan = K2FOV.test_point(ra,dec)
		return (ccd.module == mod) and (chan == channel)
	return _filter


def getObjects(ccd):
	'''input arg "channel" now refers to an index 0-3'''
	channel = Field.get_channels(ccd.module)[ccd.channel]
	objs = search.search_proposal("GO", ccd.campaign)
	#printv("Found {} objects from VJ database".format(len(objs)))
	n_objs = len(objs)
	objs.update(search.search_file(data.FULL_TARGET_LIST, ccd.campaign))
	#printv("Found {} objects from GO_all_campaigns_to_date.csv".format(len(objs)-n_objs))
	n_objs = len(objs)
	objs = dict(filter(moduleFilter(ccd), objs.iteritems()))
	#printv("Removed {} objects from outside the ccd region".format(n_objs-len(objs)))
	return objs

class BasicFitsContainer(object):

	ftypes = ('PIX','VJ','EV') # Which fits file format we're working with
	constructors = {}

	def __new__(cls, *args, **kwargs):

		if not cls.constructors:
			cls.constructors.update(dict(zip(cls.ftypes, (cls.load_target_pixels, cls.load_vj_lc, cls.load_everest_lc))))
		#cls.__new__ = super(BasicFitsContainer, cls).__new__
		cls.__new__ = cls.default_new
		return cls.__new__(cls)

	@classmethod
	def default_new(cls, *args, **kwargs):
		'''the default __new__ method for instantiation'''

		return object.__new__(cls)

	def __init__(self, ftype, *args, **kwargs):

		if ftype is None:
			self.__dict__.update(kwargs)
		else:
			assert ftype in self.ftypes, "Invalid File Type"
			constructor = self.constructors[ftype]
			self.__dict__.update(constructor(self, *args, **kwargs))

	def load_vj_lc(self, hdu, *fields):
		'''load a Vanderburg Johnson LC'''

		kwargs = {}
		for field in fields:
			kwargs.update({field:np.array(copy(hdu[1].data[field]))})
		kwargs['t'] = np.array(copy(hdu[1].data['T']))
		kwargs['EPIC'] = copy(hdu[0].header['KEPLERID'])
		kwargs['processing'] = 'VJ'
		del hdu[1].data
		#del hdu[0].header
		del hdu
		return kwargs

	def load_everest_lc(self, hdu, *fields):
		'''load an EVEREST lc'''

		kwargs = {}
		for field in fields:
			kwargs.update({field:np.array(copy(hdu[1].data[field]))})
		kwargs['t'] = np.array(copy(hdu[1].data['TIME']))
		kwargs['EPIC'] = copy(hdu[0].header['KEPLERID'])
		kwargs['processing'] = 'EVEREST'
		del hdu[1].data
		#del hdu[0].header
		del hdu
		return kwargs

	def load_target_pixels(self, hdu, *fields):
		'''Instantiate a target pixel container'''

		kwargs = {}
		for field in fields:
			kwargs.update({field:np.array(copy(hdu[1].data[field]))})
		kwargs['pixels'] = np.array(copy(hdu[1].data['FLUX']))
		kwargs['m'], kwargs['n'] = kwargs['pixels'][0].shape #WARNING! I swapped n and m, make sure that is right (it is)
		kwargs['col'] = copy(hdu[1].header['1CRV9P'])
		kwargs['row'] = copy(hdu[1].header['2CRV9P'])
		kwargs['EPIC'] = copy(hdu[0].header['KEPLERID'])
		del hdu[1].data
		#del hdu[0].header
		del hdu
		return kwargs

class PixelMapContainer:

	def __init__(self, ccd, bar = None, init=True, generator=False):
		self.ccd = ccd
		self.containers = []
		self.exclusions = []
		self.objs = {}
		self.epic_map = {}
		self.isgenerator = generator
		if init:
			self.objs.update(getObjects(ccd))
			if not self.isgenerator:
				self.load(bar=bar)

	def exclude_epic(self, *EPICs):

		for EPIC in EPICs:
			if EPIC not in self.epic_map:
				raise KeyError("{} not in PixMapContainer".format(EPIC))

			container = self.epic_map[EPIC]
			index = self.containers.index(container)
			self.exclusions.append(self.containers.pop(index))

	def include_epic(self, *EPICs):

		if not len(EPICs):
			return 0

		if EPICs[0] == 'all':
			return self.include_epic(*[cont.EPIC for cont in self.exclusions])

		excludable = [cont.EPIC for cont in self.exclusions]
		for EPIC in EPICs:
			if EPIC not in excludable:
				raise KeyError("{} not in PixMapContainer exclusions".format(EPIC))
			index = excludable.index(EPIC)
			self.containers.append(self.exclusions.pop(index))
			excludable.pop(index)

		return len(excludable)

	def __in__(self, other):

		return other in self.epic_map


	def load(self, *fields, **kwargs):

		bar = None
		failures = []
		if 'bar' in kwargs:
			bar = kwargs['bar']
		epics = self.objs.iterkeys
		hdus = fits_downloader([tpf(epic, self.ccd.campaign) for epic in epics()])
		for i, (epic, hdu) in enumerate(zip(epics(), hdus)):
			if bar is not None:
				bar.update_bar(1.0*i/len(self.objs.keys()))
			if hdu is None:
				warnings.warn("Target Pixels for 'EPIC {}' failed to download sucessfully.".format(epic))
				continue
			self.containers.append(BasicFitsContainer('PIX', hdu, self.ccd.field, *fields))
			hdu.close()
			del hdu
			gc.collect()
		self.epic_map = {container.EPIC:container for container in self.containers}

	def __getitem__(self, i):

		if type(i) is int:
			return self.containers[i]

		return self.epic_map[i]

	def __len__(self):
		return len(self.epic_map.keys())

	def __iter__(self):
		if self.isgenerator:
			epics = self.objs.iterkeys
			hdus = fits_downloader([tpf(epic, self.ccd.campaign) for epic in epics()])
			for hdu in hdus:
				yield BasicFitsContainer('PIX', hdu, self.ccd.field)
				hdu.close()
				del hdu
				gc.collect()
		else:
			for c in self.containers:
				yield c

	def save(self, hdf5_file, doc=None, clobber=False):

		#loader = LoadingBar()
		#loader.set_as_bar()
		mode = 'a'
		if not os.path.isfile(hdf5_file):
			mode = 'w'
		group_name = make_pixel_map_entry(self.ccd)
		with h5py.File(hdf5_file, mode) as f:
			if doc is None:
				doc = self.__doc__
			if doc is not None:
				if mode == 'a':
					f.attrs['/'.join((group_name, 'doc'))] = doc
				else:
					f.create_dataset('/'.join((group_name,'doc')), data=doc)
			print("  Writing... (Do Not Turn off Device or Stop Kernel)\n")
			excludable = [cont.EPIC for cont in self.exclusions]
			self.include_epic(*excludable)
			for i, g in enumerate(self):
				name = "/".join((group_name, str(g.EPIC)))
				if name not in f: # maybe have an overwrite flag?
					f.create_dataset(name+'/idx', data=np.array([g.m,g.n,g.row,g.col]))
					f.create_dataset(name+'/data', data=g.pixels)
				elif clobber:
					f.attrs[name+'/idx'] = np.array([g.m, g.n, g.row, g.col])
					f.attrs[name+'/data'] = g.pixels
				#l.update_bar(i*1.0/len(self))
			#l.update_bar(1)
			self.exclude_epic(*excludable)

		return self.ccd

	@classmethod
	def from_hdf5(cls, hdf5_file, ccd):
		'''load an hdf5 file saved with .save'''
		cont = PixelMapContainer(ccd, init=False)
		directory = make_pixel_map_entry(ccd)
		with h5py.File(hdf5_file, 'r') as f:
			group = f[directory]
			if 'doc' in group:
				cont.__doc__ = group['doc']
			for epic in group:
				if epic == 'doc':
					continue
				obj = group[epic]
				kwargs = {key:value for key, value in zip(['m','n','row','col'], np.array(obj['idx']))}
				#m, n, row, col = np.array(group[epic]['idx'])
				kwargs.update({'pixels': np.array(obj['data']), 'EPIC':int(epic)})
				#pixels = np.array(group[epic]['data'])
				#hdu = BasicFitsContainer(None, pixels=pixels, m=m, n=n, row=row, col=col, EPIC=int(epic))
				hdu = BasicFitsContainer(None, **kwargs)
				cont.containers.append(hdu)
			gc.collect()

		cont.epic_map = {container.EPIC:container for container in cont}

		return cont

class PixMapGenerator:

	def __init__(self, PixContainer, cache=False):

		if type(PixContainer) is CCD:
			PixContainer = PixelMapContainer(ccd)
		self.buff = np.zeros((1150,1150), dtype=np.float64)
		self.containers = PixContainer
		self.ccd = PixContainer.ccd
		self.stats = None
		self.cache = cache # should we cache returned items
		N = len(self.containers[0].pixels)
		self.N = N
		if cache:
			self.pixel_distribution_buffered = [False]*N
			self.pixel_distribution = [0]*N

	def __iter__(self):
		for i in xrange(self.N):
			yield self[i]

	def __len__(self):
		return len(self.containers)


	def build_stats_cache(self):
		self.stats = {'std':[],'med':[],'mean':[]}
		for c in self.containers:
			data = copy(c.pixels*1.0)
			if DATA == 'RAW_CNTS':
				data[data <= 0.5] = np.nan
			data = np.log10(data)
			self.stats["std"].append(np.nanstd(data, axis=0))
			#print "D0, std, std",data[0], self.stats['std'][-1], np.nanstd(data, axis=0)
			self.stats["med"].append(np.nanmedian(data, axis=0))
			self.stats["mean"].append(np.nanmean(data, axis=0))
			del data

	def get_zscore(self, i):
		if self.stats is None:
			self.build_stats_cache()
		self.buff[::] = np.nan
		for j,c in enumerate(self.containers): #Maybe I can do this in parallel?
			data = np.log10(copy(c.pixels[i])*1.0)
			if DATA == 'RAW_CNTS':
				data[np.isnan(data)] = self.stats['med'][j][np.isnan(data)]
				data[~np.isfinite(data)] = self.stats['med'][j][~np.isfinite(data)]
			frame = (data - self.stats['med'][j])/self.stats['std'][j]
			self.buff[c.row:c.row+c.m,c.col:c.col+c.n] = frame
		del frame
		del data
		return self.buff

	def get_unordered(self,i):
		if self.cache:
			if not self.pixel_distribution_buffered[i]:
				self.pixel_distribution[i] = np.concatenate([c.pixels[i][~np.isnan(c.pixels[i])].flat for c in self.containers])
				self.pixel_distribution_buffered[i] = True
			return self.pixel_distribution[i]
		else:
			return np.concatenate([c.pixels[i][~np.isnan(c.pixels[i])].flat for c in self.containers])


	def __getitem__(self, i):
		self.buff[::] = np.nan
		for c in self.containers: #Maybe I can do this in parallel?
			try:
				self.buff[c.row:c.row+c.m,c.col:c.col+c.n] = c.pixels[i]
			except Exception as e:
				print e
				print "ERROR"
				print c.row+c.m, c.col+c.n
		return self.buff


def bin_pixel_buffer(buff, new_shape):
	m, n = new_shape
	r, c = buff.shape
	n_row = int(r/m)
	n_col = int(c/n)
	new_buff = np.zeros(new_shape)
	for i in xrange(m):
		for j in xrange(n):
			new_buff[i, j] = np.nanmedian(buff[i*n_row:(i+1)*n_row, j*n_col: (j+1)*n_col])
	return new_buff

def make_pixel_map_entry(ccd):
	'''pixel maps are indexed by 
		campaign
		module
		channel
	'''
	mo, ch, _, ca = ccd
	return "{}/{}/{}".format(ca, mo, ch)


