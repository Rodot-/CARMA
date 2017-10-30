from utilities import downloader, queries, K2FOV
from astropy.io import fits
import numpy as np
import datetime
import time
import sys
import thread
from astropy.coordinates import SkyCoord
import astropy.units as u

class LoadingBar:

	def __init__(self):

		self.is_running = True
		#self.mapping = '_.-^-.'
		self.mapping = '-/|\\'
		self.N = len(self.mapping)
		self.n = self.N*2
		self.n = 1
		self.blank = (' '*(self.n+2)).join('\r\r')
		self.done = False

	def print_loading_bar(self):

		N = self.N
		n = self.n
		while self.is_running:
			for j in xrange(N):
				sys.stdout.write(self.blank)
				for i in xrange(j,j+n):
					sys.stdout.write('('+self.mapping[i%N]+')')
				sys.stdout.flush()
				if not self.is_running:
					self.done = True
					break
				time.sleep(0.1)
		self.done=True

	def __enter__(self):
	
		thread.start_new_thread(self.print_loading_bar, ())

	def __exit__(self, *args):
		
		N = self.N
		n = self.n
		self.is_running = False
		while not self.done:
			time.sleep(0.001)
		sys.stdout.write(self.blank)
		sys.stdout.flush()
	
DEBUG=0
EPIC=220176624
N_TILES = 2
if sys.argv[1:]:
	N_TILES = int(sys.argv[1]) - 1
	if '-e' in sys.argv:
		EPIC = int(sys.argv[sys.argv.index('-e')+1])
	if '-d' in sys.argv:
		DEBUG=1

if not DEBUG:
	sys.stderr = open('/dev/null','w')

print "Will Plot %i Objects" % N_TILES
print "Will Look at Objects Close to EPIC%i" % EPIC

print "Loading MQC..."
hdu = fits.open('data/GTR-ADM-QSO-master-sweeps-Feb5-2016.zspec.fits')
objects = np.asarray(hdu[1].data)
coords = zip(objects['RA'], objects['DEC'])
campaign = 8

FOV = K2FOV.Field(campaign)
k2 = queries.QueryK2()

print "Matching MQC to Kepler K2 Campaign %i..." % campaign
with LoadingBar():
	ra, dec = zip(*[(r,d) for r,d in coords if FOV.test_point(r,d)])
print "  Matched %i objects out of %i" % (len(ra), len(coords))

coords = zip(ra,dec)
query=[]

def filter_rows(row):

	return row and row[0].startswith('2')
print "Querying MAST For Objects within 30\"..."
'''
print "Querying MAST For Objects within 30\"..."
for i in xrange(0,len(ra)/20,200):
	print "  Query Round %i" % (i/200+1)
	ra_q = ','.join(map(str,ra[i:i+200]))
	dec_q = ','.join(map(str,dec[i:i+200]))
	query.extend(filter(filter_rows, k2.query_MAST(ra=ra_q, dec=dec_q, radius=0.5)))
	if len([q for q in query if q and q[0].startswith('2')]) > N_TILES:
		print "Found %i objects with data out of %i queried" % (N_TILES, i+200)
		break
'''
def getClosestObjects(EPIC):

	query = filter(filter_rows, k2.query_MAST(ktc_k2_id=str(EPIC)))
	q = query[0]
	ra, dec = q[4:6]
	print "Getting Objects Close to", ra, dec
	q_coord = SkyCoord(' '.join((ra, dec)), unit=('hourangle','deg')) 
	q_coord = (q_coord.ra.degree, q_coord.dec.degree)
	close_coords = []
	for i, (r, d) in enumerate(coords):
		#c = SkyCoord(r, d, unit=('deg','deg'))	
		sep = np.sqrt((r-q_coord[0])**2+(d-q_coord[1])**2)
		#sep = q_coord.separation(c)
		#if q_coord.separation(c).is_within_bounds('0d','0.5d'):
		#if sep.degree < 0.5:
		if sep < 0.5:
			close_coords.append((r, d, sep))
			#close_coords.append((r, d, sep.degree))
	print "  Found %i objects out of %i tested..." % (len(close_coords), i)
	query=[]
	close_coords = sorted(close_coords, key=lambda x: x[2])
	good_coords = []
	with LoadingBar():
		print "  Running MAST queries for objects..."
		for r, d, _ in close_coords:
			l_before = len(query)
			query.extend(filter(filter_rows, k2.query_MAST(ra=r, dec=d, radius=0.5/60)))
			l_after = len(query)
			if l_after > l_before:
				if l_after-l_before != 1:
					print "warning: multiple objects found close by"
				good_coords.append((r,d))
			if len(query) > N_TILES:
					print "  Found Enough Objects"
					break
		else:
			print "  Did not Find Enough Objects: %i..." % len(query)
			exit(1)
	return query, good_coords

#EPICs=[220167582,220169790,220176624,220220757]
#query.extend(filter(filter_rows, k2.query_MAST(ktc_k2_id=','.join(map(str, EPICs)))))

q_, close_coords = getClosestObjects(EPIC)
query.extend(q_)
N_TILES=len(query)
#exit()
EPICs = []
Dates = []
#coords = []
for q in query:
	EPICs.append(int(q[0]))
	start, end = q[8:10]
	start = datetime.datetime.strptime(start, '%Y-%m-%d %X')
	end = datetime.datetime.strptime(end, '%Y-%m-%d %X')
	delta = end-start
	Dates.append([start, delta])
hdus = []
lcs = []
print "Getting Target Pixels..."
APERTURES = ['CIRC_APER0','CIRC_APER9']


for EPIC in EPICs[:N_TILES]:
	hdus.append(np.asarray(downloader.download_target_pixels(EPIC, campaign)[1].data))

	LC = downloader.downloadVJ(EPIC, campaign)
	lcs.append([np.asarray(LC[APER].data) for APER in APERTURES])

def match_times(d1, d2):
	'''d1 is the target pixel file
		d2 is the VJ lightcurve file'''

	t1 = d1['TIME']
	t2 = d2['T']
	mask = np.in1d(t1, t2)
	return mask

masks = []
for i, (h,l,e) in enumerate(zip(hdus, lcs, EPICs)):
	
	mask = match_times(h, l[0])
	masks.append(mask)

#exit(0)



print "Plotting Target Pixels..."
from matplotlib.pyplot import subplots, show, pause, get_fignums, figure, subplot
from matplotlib import gridspec

fig1, ax1 = subplots(1,1)
ax1.scatter(*zip(*[c[0:2] for c in close_coords[:N_TILES]]), s=4, color='r')
ax1.scatter(*zip(*[c[0:2] for c in close_coords[:1]]), s=8, color='b')
FOV.plot_field(ax1)

#fig, ax = subplots(N_TILES*2,N_TILES, facecolor='#CCCCCC', figsize=[16,16])
fig = figure(facecolor='#CCCCCC', figsize=[16,16])
gs = gridspec.GridSpec(N_TILES, 2, width_ratios=[1,6])
ax = np.array(map(subplot, gs)).reshape(N_TILES, 2)
for i, a in enumerate(ax[1:,1]):
	a.get_shared_x_axes().join(a, ax[i,1])
	
#fig.tight_layout()
fig.subplots_adjust(hspace=0.15, wspace=0.01, left=0, right=1.0, bottom=0, top=0.9)
data = []
lcs_data = []
times = []
DATA = 'FLUX'

N = len(hdus[0]['FLUX'])

def update_target_pixels(ax, data, hdus, times, EPICs, lcs, lcs_data, masks, i):
		tt = 0
		was_not_masked = []
		for a, d, hdu, t, e, m in zip(ax.flat[::2], data, hdus, times, EPICs, masks):
			if hdu[DATA].shape[0] > i:
				d.set_data(hdu[DATA][i])
				tt = t[i]
			if m[i]:
				was_not_masked.append(sum(m[:i]))
			else:
				was_not_masked.append(False)
			a.draw_artist(a.set_title('EPIC ' + str(e)))
			a.draw_artist(a.patch)
			a.draw_artist(d)
		for a, data, lc, wnm in zip(ax.flat[1::2], lcs_data, lcs, was_not_masked):
			if wnm:
				i = wnm
				d = data[-1]
				l = lc[0]
				d.set_xdata([l['T'][i]])
				#for d, l in zip(data, lc)[2:]:
				#	d.set_data(l['T'], l['FCOR'])
			a.draw_artist(a.patch)
			for d in data:
				a.draw_artist(d) 
		return tt

for a,hdu,t,e in zip(ax.flat[::2], hdus, Dates, EPICs):
	a.axis('off')
	mean_pixel = np.nanmean(np.nanmax(np.nanmax(hdu[DATA], axis=1), axis=1))
	std_pixel = np.nanstd(hdu[DATA])
	max_pixel = mean_pixel + 3 * std_pixel
	min_pixel = np.nanmin(hdu[DATA])
	data.append(a.imshow(hdu[DATA][0], cmap = 'binary', vmin=min_pixel, vmax=max_pixel))
	a.set_title('EPIC ' + str(e))
	ts = hdu['TIME']
	t_delta = ts-ts[~np.isnan(ts)].min()
	while any(np.isnan(t_delta)):
		t_delta[np.isnan(t_delta)] = t_delta[np.where(np.isnan(t_delta))[0]-1]
	times.append(map(lambda x: str(t[0]+datetime.timedelta(x)), t_delta))

for a,lc in zip(ax.flat[1::2], lcs):
	lcs_data.append([])
	for l,c in zip(lc, 'kr'):
		lcs_data[-1].append(a.plot(l['T'], l['FCOR'], color=c, \
									lw=1, marker='.', ls=' ', ms=1)[0])
	lcs_data[-1].append(a.axvline(lc[0]['T'][0]))
	lc = lc[0]
	a.set_xlim(lc['T'][0], lc['T'][-1])
	std = np.std(lc['FCOR']) # limit to things within 3 std for better viewing
	mean = np.mean(lc['FCOR'])
	temp = lc['FCOR'][np.abs(lc['FCOR']) < 3*std+mean]
	ylim = (min(temp), max(temp))
	y_range = ylim[1] - ylim[0]
	center = (ylim[1] + ylim[0])/2.0
	a.set_ylim(center - y_range*0.75, center+y_range*0.75)

show(False)
pause(0.1)
print "Animating..."
while get_fignums():
	for i in xrange(N):
		if not get_fignums():
			break
		fig.draw_artist(fig.patch)
		tt = 0
		if np.isnan(hdus[0]['TIME'][i]):
			continue
		tt = update_target_pixels(ax, data, hdus, times, EPICs, lcs, lcs_data, masks, i)
		fig.suptitle('%s' % tt)
		fig.draw_artist(fig._suptitle)
		fig.canvas.update()
		fig.canvas.flush_events()
		#time.sleep(0.1)	

print "Done"
