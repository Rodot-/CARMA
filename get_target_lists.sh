#!/bin/bash
# 
# Run to update the K2 target lists by downloading
# the new list from the K2 website.

usage() {

	echo 'Usage: ./get_target_list.sh [OPTION]...'
	echo 'Download or update a full Kepler K2 target list file.'
	echo 'Example: ./get_target_list.sh -u -f K2_target_list.csv'
	echo ''
	echo 'Options:'
	echo '  -a, --all             Apply full default update, equivalent to [-d -u -c]'
	echo '  -d, --download        Download a new K2 target list from NASA website'
	echo '  -c, --clobber         Overwrite existing target list file'
	echo '  -u, --update          Update a K2 target list file by querying MAST'
	echo '  -f, --file FILENAME   Apply the update to a specifc target list file'
	echo '                            default: data/GO_all_campaigns_to_date.csv'
	echo '  -o, --out FILENAME    Set the name of the output updated file'
	echo '                            default: data/GO_all_campaigns_to_date_extra.csv'
	echo '  -h, --help            Display usage message'
}

ARCHIVE=https://keplerscience.arc.nasa.gov/data/GO_all_campaigns_to_date.csv
DEST=data
FILE=$DEST/GO_all_campaigns_to_date.csv
DEFAULTFILE=$DEST/GO_all_campaigns_to_date.csv

if [ $# -eq 0 ]; then
	echo 'Usage: ./get_target_list.sh [OPTION]...'
	echo 'Try '"'"'./get_target_list.sh --help'"'"' for more information'
fi

while [ "$1" != "" ]; do
	case $1 in
		# Do the default behavior of downloading and updating everything
		-a | --all )		DOWNLOAD=1
							UPDATE=1
							CLOBBER=1
							FILE=$DEFAULTFILE
							OUT=""
							break
							;;
		# Download the raw target list and put it into the 
		# data/ directory
		-d | --download )	DOWNLOAD=1
							;;
		# Update the contents of the local file
		-u | --update )		UPDATE=1
							;;
		# Set the optional output file
		-o | --out )		shift
							OUT="-o $1"
							;;
		# Apply update to specific file
		-f | --file )		shift
							FILE=$1
							;;
		# Overwrite $FILE
		-c | --clobber)		CLOBBER=1
							;;
		# Print help message
		-h | --help ) 		usage
							exit
							;;
		* )					usage
							exit 1
	esac
	shift
done

if [ "$DOWNLOAD" != "" ]; then
	if [ "$CLOBBER" != "" ]; then
		rm $DEFAULTFILE
	fi
	wget "$ARCHIVE" -P "$DEST"
	sed -i -e 's/, /,/g' $DEFAULTFILE
fi

if [ "$UPDATE" != "" ]; then
	python update_target_list.py $FILE $OUT
fi

