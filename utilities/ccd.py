'''a class and some methods for working with and managing individual k2 ccds in a neat way'''

import os
from collections import namedtuple
from queries.search import search_file, search_proposal

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
TARGET_LIST = os.path.join(BASE_PATH, '../data/GO_all_campaigns_to_date_extra.csv')

CCD = namedtuple('CCD', ['module', 'channel', 'field', 'campaign'], verbose=False)

def module_filter(ccd):
	K2FOV = Field(ccd.campaign)
	channel = Field.get_channels(ccd.module)[ccd.channel]
	def _filter(item):
		ra, dec = item[1][1:]
		mod, chan = K2FOV.test_point(ra,dec)
		return (ccd.module == mod) and (chan == channel)
	return _filter


def getObjects(ccd):
	'''input arg "channel" now refers to an index 0-3'''
	channel = Field.get_channels(ccd.module)[ccd.channel]
	objs = searchProposal("GO", ccd.campaign)
	objs.update(search("../data/GO_all_campaigns_to_date_extra.csv"))
	objs = dict(filter(moduleFilter(ccd), objs.iteritems()))
	return objs



