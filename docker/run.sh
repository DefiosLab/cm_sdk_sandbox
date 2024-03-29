#!/bin/bash

IMG=cm_sdk:1

echo "Hello CM_SDK!!"

echo "Container Name:"
read NAME

xhost +

FILE="/dev/video0"
if [ -e $FILE ]; then
    sudo docker run \
	     --device /dev/dri:/dev/dri \
	     --device /dev/fb0:/dev/fb0 \
	     --device /dev/tty0:/dev/tty0:mwr \
	     --device /dev/tty1:/dev/tty1:mwr \
	     --device /dev/tty2:/dev/tty2:mwr \
	     --device /dev/tty3:/dev/tty3:mwr \
	     --device /dev/video0:/dev/video0:mwr \
	     -e DISPLAY=$DISPLAY \
	     -v /tmp/.X11-unix/:/tmp/.X11-unix \
	     -v ${HOME}:${HOME} \
	     -e "HOME=${HOME}" \
	     --workdir=`pwd` \
	     -it -d --rm\
	     --name $NAME $IMG /bin/bash
else
    sudo docker run \
	     --device /dev/dri:/dev/dri \
	     --device /dev/fb0:/dev/fb0 \
	     --device /dev/tty0:/dev/tty0:mwr \
	     --device /dev/tty1:/dev/tty1:mwr \
	     --device /dev/tty2:/dev/tty2:mwr \
	     --device /dev/tty3:/dev/tty3:mwr \
	     -e DISPLAY=$DISPLAY \
	     -v /tmp/.X11-unix/:/tmp/.X11-unix \
	     -v ${HOME}:${HOME} \
	     -e "HOME=${HOME}" \
	     --workdir=`pwd` \
	     -it -d --rm\
	     --name $NAME $IMG /bin/bash    
fi


sudo docker exec -it $NAME bash
