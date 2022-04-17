FROM ubuntu:20.04

RUN echo "CM_SDK"

ENV DEBIAN_FRONTEND=noninteractive

###################
# APT
###################
RUN apt-get update && \
    # apt-get upgrade && \
    apt-get install sudo curl wget zip neovim -y
  
RUN apt-get install build-essential make g++ gcc-multilib g++-multilib cmake -y
RUN apt-get install clinfo ocl-icd-opencl-dev ocl-icd-libopencl1 -y
RUN apt-get install libffi7 libffi-dev -y

###################
# ADD: HOME DIR
###################
RUN useradd -m worker
WORKDIR /home/worker

###################
# CM_SDK
###################
RUN wget https://01.org/sites/default/files/downloads//cmsdk20211028.zip
RUN unzip cmsdk*.zip
RUN mv cm_sdk*/ cm_sdk/

WORKDIR cm_sdk/
ENV ROOT_PATH=/home/worker/cm_sdk/
ENV CSDK_DIR=$ROOT_PATH
ENV CM_INCLUDE_DIR=$CSDK_DIR/usr/include
ENV OPENCL_VENDOR_PATH=$CSDK_DIR/etc/OpenCL/vendors
ENV PATH=$CSDK_DIR/usr/bin:$CSDK_DIR/usr/local/bin:$PATH
ENV LD_LIBRARY_PATH=$CSDK_DIR/usr/lib:$CSDK_DIR/usr/lib/x86_64-linux-gnu/:$CSDK_DIR/usr/local/lib/intel-opencl:$CSDK_DIR/usr/local/lib

###################
# Level0
###################
RUN apt-get install -y gpg-agent wget software-properties-common
RUN wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | sudo apt-key add -
RUN apt-add-repository 'deb [arch=amd64] https://repositories.intel.com/graphics/ubuntu focal main'

RUN apt-get install -y intel-level-zero-gpu level-zero level-zero-dev


ENV TZ=Asia/Tokyo