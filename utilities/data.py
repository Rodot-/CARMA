'''data.py
a module for managing the locations of various 
data files that the module relies upon
'''

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../data')

K2_FOOTPRINT = os.path.join(DATA_DIR, 'k2-footprint.csv')
PIXEL_MAP_DIR = os.path.join(DATA_DIR, 'PixelMaps')
FULL_TARGET_LIST = os.path.join(DATA_DIR, 'GO_all_campaigns_to_date_extra.csv')
