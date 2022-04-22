OPENCVHEADER = /usr/local/include/opencv4
LIB_LOADPATH = -L/usr/local/lib
LINK_OPENCV_LIBS = \
-lopencv_core \
-lopencv_imgcodecs \
-lopencv_highgui \
-lopencv_imgproc

LDFLAGS = -lm -lstdc++ $(LIB_LOADPATH) $(LINK_OPENCV_LIBS)

all: 
	${CSDK_DIR}/usr/bin/cmc -emit-spirv -fcmocl -mcpu=SKL -m64 \
				-isystem ${CSDK_DIR}/usr/include kernel.cpp -o kernel.spv.skl

	g++ -m64 -DKERNEL=\"kernel.spv.skl\" \
		-I${CSDK_DIR}/usr/local/include -L${CSDK_DIR}/usr/local/lib  \
		-Wl,-rpath -Wl,${CSDK_DIR}/usr/local/lib -std=c++17 -DCM_COMPILE_SPIRV\
		host_l0.cpp -lze_loader -o vector.l0.skl \
		-I$(OPENCVHEADER) $(LIB_LOADPATH) $(LINK_OPENCV_LIBS) 


# shim: 
# 	g++ -std=gnu++17 -DCMRT_EMU -DSHIM -shared -fpic \
# 				-I${CSDK_DIR}/usr/include/libcm -I${CSDK_DIR}/usr/include/shim \
# 				-L${CSDK_DIR}/usr/lib -Wl,-rpath -Wl,${CSDK_DIR}/usr/lib \
# 				kernel.cpp -o kernel.l0.shim -lcm

# 	g++ -std=gnu++17 -DKERNEL=\"kernel.l0.shim\" \
# 			-I${CSDK_DIR}/usr/include/libcm -I${CSDK_DIR}/usr/include \
# 			-L${CSDK_DIR}/usr/lib -Wl,-rpath -Wl,${CSDK_DIR}/usr/lib \
# 			host_l0.cpp -lshim_l0 -ligfxcmrt_emu -lcm -Wl,--disable-new-dtags -o vector.l0.shim

clean:
	rm *.skl tmp* *.shim
