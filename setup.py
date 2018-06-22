'''
Not currently a real setup utility, just makes the correct directories
'''
import os

def mkdir(path):

	if not os.path.isdir(path):
		os.mkdir(path)

for path in ('data','data/PixelMaps','plots'):
	mkdir(path)

