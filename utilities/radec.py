#converts ra and dec from hms,dms to deg and back
'''
Cline Args
	-f <file> [<file2> ...] | read in file(s) line by line
	-v | verbose
	-t | output in table format
	-s | output only sexigesimal coordinates
	-d | output only degree coordinates
'''
from __future__ import print_function
import sys
import os
import re

INIT=True #Header Initialization for output
VERBOSE=False
SEXIGESIMAL = True
DEGREE = True

class OutOfBoundsError(Exception):
	pass

class BadPatternError(Exception):
	pass

def printE(*args):
	if VERBOSE:	print(*args, file=sys.stderr)

def bootstrapUnitConversion(pack, conversion):

	array, total = pack
	new_value = int(1.0*total/conversion)
	new_total = total - new_value*conversion
	array.append(new_value)
	return (array, new_total)

def unitConversion(value, units):

	def convert(value, units):
		array = []
		pack = (array, value)
		reduce(bootstrapUnitConversion, (pack,)+ unitConversion.factors[units])
		return array
	unitConversion = convert
	unitConversion.factors = {
		'hms':(
			15.0, #Deg to Hours
			1/4.0, #Deg to Hour Minutes
			1/240.0, #Deg to Hour Seconds
			1/240000.0 #Deg to Hour milliseconds
		),
		'dms':(
			1.0, #Deg to deg
			1/60.0, #Deg to Minutes
			1/3600.0, #Deg to Seconds
			1/3600000.0 #Deg to milliseconds
		)
	}
	return unitConversion(value, units)

def hmsToDeg(h, m, s):

	h, m, s = map(float, (h, m, s))
	if m >= 60 or s >= 60:
		plr = ('', 'is') # plural form
		msg = []
		if s >= 60: msg.append('seconds')
		if m >= 60:	msg.append('minutes')
		if len(msg) > 1:
			plr = ('s', 'are')
			msg = ' and '.join(msg)
		else: msg = msg[0]
		raise OutOfBoundsError(
			"{0} column{1} {2} too large in right ascension"
			.format(msg, plr[0], plr[1]))

	while h > 23:
		h -= 24
	return h*15.0+m*0.25+s/240.0

def dmsToDeg(d, m, s):

	d, m, s = map(float, (d, m, s))
	if m >= 60 or s >= 60 or d > 90:
		plr = ('', 'is') # plural form
		msg = []
		if s >= 60: msg.append('seconds')
		if m >= 60:	msg.append('minutes')
		if d >= 90: msg.append('degrees')
		if len(msg) > 1:
			plr = ('s', 'are')
			if len(msg) == 2: msg = ' and '.join(msg)
			else: msg = ', and '.join((', '.join(msg[:-1]), msg[-1]))
		else: msg = msg[0]
		raise OutOfBoundsError(
			"{0} column{1} {2} too large in declination"
			.format(msg, plr[0], plr[1]))

	return (d+m/60.0+s/3600.0)

def degTohms(deg):
	while deg > 360:
		deg -= 360
	h, m, s, ms = unitConversion(deg, 'hms')
	s += ms/1000.0
	return (h, m, s)

def degTodms(deg):
	if deg > 90:
		raise OutOfBoundsError("DEC value of out range")
	d, m, s, ms = unitConversion(deg, 'dms')
	s += ms/1000.0
	return (d, m, s)

def getSign(match):

	return '+' if match.group('s') in (None,'+') else '-'

def ra(coord, to='hms'):

	raw = Parser.parse_coord(coord, pattern='ra')
	if to == 'hms':
		return ' '.join(map(str, raw[0])) 
	elif to == 'deg':
		return raw[1]

def dec(coord, to='dms'):

	raw = Parser.parse_coord(coord, pattern='dec')
	if to == 'dms':
		return raw[0]+' '.join(map(str, raw[1])) 
	elif to == 'deg':
		return raw[2]*{'-':-1,'+':1}[raw[0]]

class Parser:
	'''Handles String Parsing and Unit Conversions'''

	# some short substitutions to make to regex string smaller
	f = "\d+(?:\.\d+)?" # floating point number
	s = ["(?:%s|:| |\.)" % x for x in 'hmd'] #seperator
	g = ["?P<%s>" % x for x in 'rh rm rs s dd dm ds'.split()] #groups

	patterns = {
		'ra': (
			r"^({0}\d+){8} *({1}\d+){9} *({2}{7})s?".format(*(g+[f]+s)),
			r"^(?P<ra>\d+(?:\.\d+)?)(?:d|deg)?"
		), 
		'dec': (
			r"({3}\+|-)?({4}\d+){10} *({5}\d+){9} *({6}{7})s?$".format(*(g+[f]+s)),
			r"(?P<s>\+|-)?(?P<dec>\d+(?:\.\d+)?)(?:d|deg)?$"
		)
	}
	patterns['combined'] = [' *'.join((ra,dec)) for ra, dec in zip(patterns['ra'], patterns['dec'])]

	@classmethod
	def parse_coord(cls, string, pattern='combined'):

		use_ra = pattern in ('ra', 'combined')
		use_dec = pattern in ('dec', 'combined')
		for i, pattern in enumerate(cls.patterns[pattern]):
			match = re.match(pattern, string)
			if match is not None:
				if use_dec:
					sign = getSign(match) # get the sign of the dec
				get = match.group # function that retrieves the regex matches
				printE("Got a match in {}".format(pattern))
				printE("regex match groups: {}".format(match.groups()))
				if use_dec:
					printE("Sign: {}, {}".format(sign, match.group('s')))
				try:
					if i == 0: # hms, dms
						if use_ra:
							ra = hmsToDeg(*get('rh','rm','rs'))
						if use_dec:
							dec = dmsToDeg(*get('dd','dm','ds'))
					elif i == 1: # Deg
						if use_ra:
							ra = float(get('ra'))
						if use_dec:
							dec = float(get('dec'))
			
					if use_ra:
						hms = degTohms(ra)
					if use_dec:
						dms = degTodms(dec)
				except OutOfBoundsError as err:
					print("OutOfBoundsError: {}".format(err), file=sys.stderr)
					return None

				break
		else:
			raise BadPatternError("Could not match input to an existing template")
		result = []
		if use_dec:
			result.append(sign)
		if use_ra:
			result.append(hms)
		if use_dec:
			result.append(dms)
		if use_ra:
			result.append(ra)
		if use_dec:
			result.append(dec)
		return tuple(result)

	@classmethod
	def parse(cls, string, pattern='combined'):

		for i, pattern in enumerate(cls.patterns[pattern]):
			match = re.match(pattern, string)
			if match is not None:
				sign = getSign(match) # get the sign of the dec
				get = match.group # function that retrieves the regex matches
				printE("Got a match in {}".format(pattern))
				printE("regex match groups: {}".format(match.groups()))
				printE("Sign: {}, {}".format(sign, match.group('s')))
				try:
					if i == 0: # hms, dms
						ra = hmsToDeg(*get('rh','rm','rs'))
						dec = dmsToDeg(*get('dd','dm','ds'))
					elif i == 1: # Deg
						ra, dec = map(float,get('ra','dec'))

					hms = degTohms(ra)
					dms = degTodms(dec)
				except OutOfBoundsError as err:
					print("OutOfBoundsError: {}".format(err), file=sys.stderr)
					return None

				break
		else:
			raise BadPatternError("Could not match input to an existing template")
		return sign, hms, dms, ra, dec
		

def parse_coordinates(string):
	'''parses an input string and converts to a coordinate in deg'''

def printTable(sign, hms, dms, ra, dec):
	global INIT

	hms = "{:0>2.0f}:{:0>2.0f}:{:0>2.2f}".format(*hms)
	dms = "{}{:0>2.0f}:{:0>2.0f}:{:0>2.1f}".format(sign, *dms)
	ra = "{:0<3.6f}".format(ra)
	dec = "{}{:0<2.7f}".format(sign, dec)

	if INIT:
		if SEXIGESIMAL and DEGREE:
			print("{},{},{},{}".format("RA","DEC","RA_hms","DEC_dms"))
		else:
			print("{},{}".format("RA","DEC"))
		INIT = False
	line = []
	if DEGREE:
		line.append("{},{}".format(ra,dec))
	if SEXIGESIMAL:
		line.append("{},{}".format(hms,dms))
	print(','.join(line))
		

def printNice(sign, hms, dms, ra, dec):
	global INIT

	hms = "{:0>2.0f}h {:0>2.0f}m {:0>2.2f}s".format(*hms)
	dms = "{}{:0>2.0f}d {:0>2.0f}m {:0>2.1f}s".format(sign, *dms)
	ra = "{:0<3.6f}d".format(ra)
	dec = "{}{:0<2.7f}d".format(sign, dec)

	if INIT:
		print('')
		print("        {0:^16}|{1:^16}".format("RA", "DEC"))
		print("        {0:-^16}+{0:-^16}".format(''))
		INIT = False
	else:
		print("        {0:^16}|{0:^16}".format(''))
	if SEXIGESIMAL:
		print("        {0:^16}|{1:^16}".format(hms, dms))
	if DEGREE:
		print("        {0:^16}|{1:^16}".format(ra, dec))

printFunc = printNice #by default, what function we use to print results

def main(argc,argv): #just return the format
	global VERBOSE, INIT, SEXIGESIMAL, DEGREE, printFunc
	
	if not argc:
		print("Please Provide a Pair of J2000 Coordinates")
		return 1
	assert ('-s' not in argv) if ('-d' in argv) else True,\
		"Please Select only 1 Coordinate Representation"
	if '-t' in argv: #Table
		argv.remove('-t')
		printFunc = printTable
	if '-s' in argv: #Sexigesimal
		argv.remove('-s')
		DEGREE=False
	if '-d' in argv: #Degree
		argv.remove('-d')
		SEXIGESIMAL=False
	if '-v' in argv: #Verbose
		VERBOSE = True
		argv.remove('-v')
	if '-f' in argv: #File
		argv.remove('-f')
		files = argv
		if not files:
			files = [sys.stdin]
		errors = []
		printE("Got Cline Arg '-f', Reading in File{}".format('s' if len(argv)-1 else ''))
		for filename in files:
			if filename is sys.stdin:
				f = filename
			elif os.path.isfile(filename):
				f = open(filename,'r')
			else:
				print("Error: File '{}' Not Found".format(filename))
				continue
			INIT = True
			if printFunc != printTable:
				print("Results from file: '{}'".format(filename))
			for i,line in enumerate(f):
				line = line.strip()
				argv = line.split()
				argc = len(line)
				if argc:
					try:
						main(argc, argv)
					except Exception as err:
						name = err.__repr__()
						msg = "{}".format(name)
						errors.append((filename, i, line, msg))
			if printFunc != printTable:
				print('')
			
		if errors:
			print("Errors Were Encountered in the Following Files:", file=sys.stderr)
		error_files = {error[0] for error in errors}
		error_info = {f:[e[1:] for e in errors if e[0] == f] for f in error_files} 
		for error_file in error_files:
			print(" '{}'".format(error_file), file=sys.stderr)
			for info in error_info[error_file]:
				printE("  in line {}: '{}'\n   {}".format(*info))
		
		return 1

	string = ' '.join(argv).strip()
	printE("Processing this string: {0}".format(string))

	result = Parser.parse(string)
	if result is not None: printFunc(*result)

	return 1

def test():
	global INIT, DEGREE, SEXIGESIMAL, printFunc
	import glob
	testfiles = ' '.join(glob.glob("../*.dat"))
	trials = ['345.12236 52.6543',
		'345.12236d +52.6543deg',
		'345.12236deg -52.6543d',
		'12 54 12.5422 23 12 23.654',
		'12 54 12.5422 +23 12 23.654',
		'12 54 12.5422 -23 12 23.654',
		'',
		'25 34 35.123 +34 12 46.23', #out of bounds things that are handled
		'378.6452 -89.4563',
		'236.235 -92.654', #out of bounds things that are not
		'72.7354 91.75',
		'23 65 24.65 -65 12 54.24',
		'21 45 89 +72 12 23',
		'8 756 92.5 -66 23 12.777',
		'12 34 12.3 -90 32 45.7',
		'12 35 54.7 90 66 82.2',
		'12 12.5 34.156 40 12 12.2', #Bad syntax	
		'04 23 15.3deg 12 23 34.55',
		'345.235h -22.35d',
		'-f {}'.format(testfiles), #read files
		'-f -s {}'.format(testfiles), #Only Sexigesimal
		'-f -d {}'.format(testfiles), #Only Degree
		'-f -s -d {}'.format(testfiles), #Error
		'-f -s -t {}'.format(testfiles), #Table
		'-f -t {}'.format(testfiles) #Table
	]
	for i, trial in enumerate(trials):
		INIT = True
		DEGREE = True
		SEXIGESIMAL = True	
		printFunc = printNice
		print("{:<3} Running Trial with input: {}".format(str(i)+'.', trial))
		argv = trial.split()
		argc = len(argv)
		if trial == '': print("  Testing No Input")
		else:
			if i < 6:
				if i < 3: print("  Testing RA (deg), DEC (deg)", end='')
				else: print("  Testing RA (hms), DEC (dms)", end='')
				if '+' in trial: print(" With '+' sign DEC")
				elif '-' in trial: print(" With '-' sign DEC")
				else: print(" With no sign DEC")
		if i > 5:
			print("  Testing Errors")
		try:
			main(argc, argv)
		except BadPatternError as err:
			print("BadPatternError: {}".format(err))
		except AssertionError as err:
			print("AssertionError: {}".format(err))
		print('')

if __name__ == '__main__':
	argv = sys.argv[1:]
	argc = len(argv)
	if '--test' in argv:
		print("Running Tests")
		if '-v' in argv: VERBOSE = True
		test()
	elif '--help' in argv:
		print(__doc__)
		sys.exit(1)
	else:
		try:
			main(argc, argv)
		except BadPatternError as err:
			print("BadPatternError: {}".format(err))
