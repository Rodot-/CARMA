'''general tools for searching campaigns'''
import csv
import urllib2
import re

def search_proposal(Proposal, campaign = 8): 
	'''Seach a proposal number for EPIC IDs, 
	returns a dict with EPIC IDs as keys to KepMag and RA and DEC coords
	relys on database of objects with SFF processing'''	
 
	socket = urllib2.urlopen('https://www.cfa.harvard.edu/~avanderb/allk2c%iobs.html' % campaign) 
	body = socket.read() 
 
	pattern = "EPIC (\d{9})</a>, Kp = (\d+\.\d+), RA = (\d+\.\d+), Dec = (-?\d+\.\d+).+%s" % Proposal 
	matches = re.findall(pattern, body) 
	socket.close() 
	EPIC = [int(i[0]) for i in matches] 
	coords = [(float(i[1]), float(i[2]), float(i[3])) for i in matches] 
	Result = dict(zip(EPIC, coords)) 
	 
	return Result 
 
def search_file(target_file, campaign=8): 
	'''Search a k2 target list file for objects in a certain campaign'''
	with open(target_file,'r') as f: 
		reader = csv.DictReader(f)
		if campaign == 91:
			campaign = str('9a')
		elif campaign == 92:
			campaign = str('9b') 
		data = dict(zip(reader.fieldnames, zip(*[[row[key].strip() for key in reader.fieldnames] for row in reader if row['campaign'] == str(campaign)])))  # the strips are a hack so I don't have to admit to myself that campaign number should be a str
	EPIC = data['EPIC ID'] 
	func = lambda x: float(x) if (x and x not in ['None',' ']) else None 
	RA = map(func, data['RA (J2000) [deg]']) 
	DEC = map(func, data['Dec (J2000) [deg]']) 
	MAG = map(func, data['magnitude']) 
	return {int(epic): (mag, ra, dec) for epic, mag, ra, dec in zip(EPIC, MAG, RA, DEC) if None not in (mag, ra, dec)} 

