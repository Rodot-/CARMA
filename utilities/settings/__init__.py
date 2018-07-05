'''
settings

A manager for constants defining general behavior of the library
'''
import os
import json

class Setting:

	def __init__(self, settings):

		self.__dict__.update(settings)
		self.settings = settings
		
SETTINGS_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(SETTINGS_DIR, 'settings.json')
LOG_DIR = os.path.join(SETTINGS_DIR, '../log')
SETTINGS = {}

def save(settings=None, settings_file=SETTINGS_FILE):
	'''save current or specific settings to the default or other settings file'''
	global SETTINGS
	if settings is None:
		settings = {key: value.settings for key, value in SETTINGS.items()}
	with open(settings_file, 'wb') as f: 
		f.write(json.dumps(settings, indent=4))

def load(settings_file=SETTINGS_FILE):
	'''load global settings from a default or custom settings file'''
	global SETTINGS

	with open(settings_file, 'rb') as f:
		SETTINGS = {key:Setting(value) for key, value in json.loads(f.read()).items()}
		globals().update(SETTINGS)


if not os.path.isfile(SETTINGS_FILE):

	default_settings = {
		'VERBOSE': 
			{'level': 0, '__doc__': """Verbosity level: int 0-2
	0: Default Python Warnings and Errors (Default)
	1: LoadingBars and Formatted Runtime info for logging
	2: In-depth run-time info for debugging"""},
		'LOGGING':
			{'level':0, 'logfile': 'k2.log', '__doc__': """Logging level: int 0-2
	0: Do not log (Default)
	1: Log Command Usage
	2: Log all verbosity information"""}
		}

	save(default_settings)

load()

