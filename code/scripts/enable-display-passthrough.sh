#!/bin/sh
# this script is needed by the selfplay docker container to allow GUIs to be displayed on the host

xhost +

if [ $? -ne 0 ]; then
	echo "Failed to enable display passthrough"
	exit 1
else 
	echo "Display passthrough enabled!"
	exit 0
fi
