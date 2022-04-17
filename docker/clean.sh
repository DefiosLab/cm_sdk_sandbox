#!/bin/bash

Sharedfile="~\/":

echo "Remove CM_SDK!!"

echo "Container Name:"
read NAME

sudo docker stop $NAME
sudo docker rm $NAME
