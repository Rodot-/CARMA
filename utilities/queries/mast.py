'''
queries.py - Query various databases for observations

10/24/2017
'''
import os
import urllib2
import urllib
import re
import textwrap
import csv
import datetime

from astropy.units import hourangle, deg
from astropy.coordinates import Longitude, Latitude

from .. import radec

#TODO: networking in context manager to handle network errors

class Query:
	'''this will eventually be a base class for query objects'''
	pass

# maps a MAST type to a useful python datatype
TYPE_MAP = {
	'integer': int,
	'string': str,
	'substring': str,
	#'ra': lambda ra: Longitude(ra, unit=hourangle),
	#'dec': lambda dec: Latitude(dec, unit=deg),
	'ra': lambda ra: radec.ra(ra, to='deg'),
	'dec': lambda dec: radec.dec(dec, to='deg'),
	'ustring': unicode,
	'datetime': lambda date: datetime.datetime.strptime(date, '%Y-%m-%d %X'),
	'float': float
} 


class Query:
	'''Will eventually be a subclass, developed independently for
	now to give an example of structure of the super class'''

	archive_url = "https://archive.stsci.edu"
	column_header = ['Column Name', 'Column Label', 'Description', \
					'Examples/Valid Values', 'Units']
	output_format = 'outputformat=CSV&'
	search_str = 'search.php?action=Search&' + output_format
	field_url = os.path.join(archive_url, 'search_fields.php?')

	def query_MAST(self, **kwargs):
		'''query MAST for a set of parameters, can be cleaned up
			Should be broken up into following methods:
				Verify Inputs/Generate URL -> return socket object or url
					(should also cache queries)
				-Then use the socket in any of the methods below
					RAW_Read -> just read the socket directly
					Export_Read -> Export to File
					Get_Data -> return a numpy array or something readable
				
		'''
		if not self.fields:
			self.fields.extend(self.__get_fields())

		# Verify the keys
		available_keys = set(row[0] for row in self.fields)
		provided_keys = set(kwargs.keys())
		N_keys = len(kwargs)
		union = provided_keys & available_keys
		#if len(union) != N_keys:
		#	print "Error: Invalid Key(s):", ', '.join(list(provided_keys - union))
		#	#exit()
		#	return
		# Verify value range (do this a bit later)

		# Format fields
		query = urllib.urlencode(kwargs)
		url = self.search_url + query
		
		socket = urllib2.urlopen(url)

		# Should end here
		#reader = csv.reader(socket, delimiter=',')
		return self.iter_read_socket(socket)

	@classmethod
	def to_table(cls, reader):
		'''converts a csv iterator to a formatted table'''
	
		header = reader.next()
		types = reader.next()
		rows = []
		for i,line in enumerate(reader):
			rows.append([TYPE_MAP[t](l) if l else None for t, l in zip(types, line)])

		return dict(zip(header, zip(*rows)))

	def iter_read_socket(self, socket):

		reader = csv.reader(socket, delimiter=',')
		for line in reader:
			yield line
		socket.close()

	def print_fields(self):
		''' print the available fields for this query'''
		# Make sure we have the data to do this
		if not self.fields:
			self.fields.extend(self.__get_fields())
		# make a nice looking table
		col_widths = [20, 14, 24, 20, 10] # Maybe make this less hard-coded
		# Maybe have it based on the lenght of the longest item in col 0?
		line_sep = '-'+'-+-'.join(['-'*w for w in col_widths])
		print line_sep
		for field in [self.column_header]+self.fields:
			# Wrap the text over the wdith
			cols = [textwrap.wrap(line, w) for line, w in zip(field, col_widths)]
			# Make sure every column has the same number of rows
			N_lines = max(map(len, cols))	
			for w, col in zip(col_widths, cols):
				# Make every column the same height by inserting blank rows
				diff = N_lines - len(col) # how many rows are needed
				idx = diff/2 # index of what should be half the rows
				new_col = [' '*(w+2)]*diff # +2 because of spaces at the ends
				# While we're at it, lets format the rows
				format_str = ' {:^%i} ' % w 
				new_col[idx:idx] = (format_str.format(row) for row in col)
				col[::] = new_col[::]
			# Print out our formatted columns
			for row in zip(*cols):
				print '|'.join(row)
			print line_sep

	def __get_fields(self):
		'''get the available fields for this query'''
		# TODO: Do this in a single Pattern Match
		# TODO: Context manager for the socket
		# request the data from the socket
		socket = urllib2.urlopen(self.field_url)
		body = socket.read

		# set up the regex parameters for the search
		flags = re.MULTILINE | re.DOTALL

		# Set up a regex to extract the table info
		field_table_regex = r'^<td.*?>(?:<i>)?(?:<a.*?>)?(.*?)(?:</a>)?(?:</i>)?</td>$'
		fields = re.findall(field_table_regex, body(), flags)

		socket.close()	

		# Reshape the matches to make rows and columns
		lines = (fields[i:i+6] for i in xrange(0,len(fields),6))

		return lines

