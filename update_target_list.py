'''get the RA and dec for objects that don't have available data in the csv file'''

import csv, sys
from utilities.queries.sky_search import epic_search
from utilities.queries import k2

CHUNK_SIZE = 500

def chunker(data, chunk_size):

	return [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

def build_table(csv_file, outfile=None):

	if outfile is None:
		outfile = csv_file.replace('.csv','_extra.csv')
	Q = k2.Query()
	with open(csv_file, 'r') as target_list:
		print(csv_file)

		reader = csv.reader(target_list)
		header = reader.next()
		# Not using the header
		# we know the columns are:
		# EPIC, RA, DEC, Kep Mag, Inverstigation ID

		rows = [[item.strip() for item in line] for line in reader]
		epic_index_map = {row[0]:i for i, row in enumerate(rows) if row[1] == ''}
		needed_objects = epic_index_map.keys()

		keys = ('K2 ID', 'RA (J2000)', 'Dec (J2000)', 'KEP Mag')
		n_new = 0

		print("Querying MAST...")
		query = Q.to_table(iter(epic_search(needed_objects)))

		if not query:
			print("  WARNING: QUERY FAILED, NO MATCHES FOUND")
			sys.exit(1)
			return

		for i, EPIC in enumerate(query[keys[0]]):
			index = epic_index_map[str(EPIC)]
			
			rows[index][1:4] = map(str, (query[keys[j+1]][i] for j in xrange(3)))

			#ra = str(query[keys[1]][i])
			#dec = str(query[keys[2]][i])
			#kp = str(query[keys[3]][i])
			
			#rows[index][1] = ra
			#rows[index][2] = dec
			#rows[index][3] = kp
	
			n_new = i

		with open(outfile,'w') as f:
			lines = [header] + rows
			lines = [",".join(line) for line in lines]
			f.write('\n'.join(lines))

		print("  New data for {} objects found".format(n_new))
		print("  Wrote out {}".format(outfile))

if __name__ == '__main__':

	outfiles = []
	if len(sys.argv) == 1:
		print("Please Select one or More Input CSV files")
	while '-o' in sys.argv:
		outfiles.append(sys.argv.pop(sys.argv.index('-o')+1))
		sys.argv.remove('-o')
	outfiles.extend([None]*len(sys.argv[1:]))
	for csv_file, outfile in zip(sys.argv[1:], outfiles):
		#try:
		build_table(csv_file, outfile)
		#except Exception as e:
		#	print(e.__repr__())

		
		

