import sys
from PIL import Image
import glob
import numpy as np
from matplotlib.pyplot import subplots, show
import re, os
from itertools import izip

TYPE = 'Full' # 'Diff'
TYPE = 'Diff'

base_dir = 'plots'
if len(sys.argv) > 1:
	base_dir = sys.argv[1]

image_files = glob.glob('Module*.png')
print(image_files[:])
#pattern = os.path.join(base_dir, 'Module(\d{1,2})Channel([0-3])PixelMap(Full|Diff).png')
pattern = str('Module(\d{1,2})Channel([0-3])PixelMap(Full|Diff).png')
engine = re.compile(pattern)
matches = (engine.match(file_name).groups() for file_name in image_files)
data = (np.array(Image.open(file_name)) for file_name in image_files)

images = ({'name':f, 'module':int(m), 'channel':int(c), 'type':t, 'data':data} \
	for f, (m, c, t), data in izip(image_files, matches, data) if t == TYPE)

row_bounds = (79,556)
col_bounds = (145, 813)

# Subimage dimensions
WIDTH = col_bounds[1] - col_bounds[0]
HEIGHT = row_bounds[1] - row_bounds[0]


# Padding
CHANNEL_PAD = 50
MODULE_PAD = 100

# Final image dimensions
WIDTH_F = (CHANNEL_PAD + WIDTH) * 10 + (10/2-1) * MODULE_PAD - CHANNEL_PAD+100
HEIGHT_F = (CHANNEL_PAD + HEIGHT) * 10 + (10/2-1) * MODULE_PAD - CHANNEL_PAD

# Define module flags
horizontal_modules = [16,17,11,12,13,6]
flipped_modules = [14,15,9,10,20,18,19,22,23,24]
zero_modules = []
last_modules = [18,19,22,23,24]

def get_coarse_grid_position(module):

	col = (module%5)-1
	row = (module - 1)/5
	print "    CP: {}, {}".format(row, col)
	return row, col

def get_fine_grid_position(coarse_position, channel, O_flag, F_flag ,l_flag):

	yp, xp = coarse_position
	pos = (channel + (O_flag + F_flag)) % 4 
	print(coarse_position,channel,pos)
	y = 2*yp +  ((pos) / 2) - (l_flag*((channel)%2))
	x = 2*xp + (1-(((pos + 1) % 4) / 2)) - (l_flag*((channel+1)%2))
	if (l_flag == 1):
		if(channel == 0):
			y = 2*yp +  0
			x = 2*xp + 1
	if (l_flag == 1): 
		if (channel == 3):
			print("TRUEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE") 
			y = 2*yp +  1
			x = 2*xp + 1		

	#y = 2*yp + (pos / 2)
	#x = 2*xp + (((pos + 1) % 4) / 2)
	print "    FP: {}, {}".format(y, x)
	return y, x

def get_true_grid_position(fine_position):

	yf, xf = fine_position
	y = yf * (HEIGHT + CHANNEL_PAD) + yf/2 * MODULE_PAD
	x = xf * (WIDTH + CHANNEL_PAD) + xf/2 * MODULE_PAD
	return y, x

def get_grid_position(image):

	cp = get_coarse_grid_position(image['module'])

	O_flag = 1 if image['module'] in horizontal_modules else 0
	O_flag = O_flag if image['module'] not in zero_modules else 1
	F_flag = 1 if image['module'] in flipped_modules else 2
	l_flag = 1 if image['module'] in last_modules else 0
	fp = get_fine_grid_position(cp, image['channel'], O_flag, F_flag, l_flag)
	y, x = get_true_grid_position(fp)

	return y, x


final_image = np.zeros((HEIGHT_F, WIDTH_F, 4), dtype=np.uint8)
final_image[:] = 155

print "Building Image..."
for image in images:
	y, x = get_grid_position(image)
	print "  Module:{}, Channel:{}, row:{}, col:{}".format(image['module'], image['channel'], y, x)
	sub_image = image['data'][row_bounds[0]:row_bounds[1], col_bounds[0]:col_bounds[1]]
	final_image[y:y+HEIGHT, x:x+WIDTH] = sub_image[::]

print "Done"

final = Image.fromarray(final_image)
final.save("K2All%sPixelMaps.png" % TYPE)

#fig, ax = subplots(1,1, figsize=(16,9), dpi=600)
#ax.imshow(final_image)
#show()

