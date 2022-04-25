OPENCVHEADER = /usr/local/include/opencv4
LIB_LOADPATH = -L/usr/local/lib
LINK_OPENCV_LIBS = \
-lopencv_core \
-lopencv_imgcodecs \
-lopencv_highgui \
-lopencv_imgproc

LDFLAGS = -lm -lstdc++ $(LIB_LOADPATH) $(LINK_OPENCV_LIBS)
all : s1 s2 s3

s1:
	${CSDK_DIR}/usr/bin/cmc -emit-spirv -fcmocl -mcpu=SKL -m64 \
				-isystem ${CSDK_DIR}/usr/include step1/kernel.cpp -o step1.spv.skl

	g++ -m64 -DKERNEL=\"step1.spv.skl\" \
		-O3 -fopenmp \
		-DBLOCK_X=1 -DBLOCK_Y=1 \
		-I${CSDK_DIR}/usr/local/include -L${CSDK_DIR}/usr/local/lib -I./include/ \
		-Wl,-rpath -Wl,${CSDK_DIR}/usr/local/lib -std=c++17 -DCM_COMPILE_SPIRV\
		host_l0.cpp src/cpu_zncc.cpp -lze_loader -o step1.l0.skl \
		-I$(OPENCVHEADER) $(LIB_LOADPATH) $(LINK_OPENCV_LIBS)
s2:
	${CSDK_DIR}/usr/bin/cmc -emit-spirv -fcmocl -mcpu=SKL -m64 \
				-isystem ${CSDK_DIR}/usr/include step2/kernel.cpp -o step2.spv.skl

	g++ -m64 -DKERNEL=\"step2.spv.skl\" \
		-O3 -fopenmp \
		-DBLOCK_X=8 -DBLOCK_Y=8 \
		-I${CSDK_DIR}/usr/local/include -L${CSDK_DIR}/usr/local/lib -I./include/ \
		-Wl,-rpath -Wl,${CSDK_DIR}/usr/local/lib -std=c++17 -DCM_COMPILE_SPIRV\
		host_l0.cpp src/cpu_zncc.cpp -lze_loader -o step2.l0.skl \
		-I$(OPENCVHEADER) $(LIB_LOADPATH) $(LINK_OPENCV_LIBS)

s3:
	${CSDK_DIR}/usr/bin/cmc -emit-spirv -fcmocl -mcpu=SKL -m64 \
				-isystem ${CSDK_DIR}/usr/include step3/kernel.cpp -o step3.spv.skl

	g++ -m64 -DKERNEL=\"step3.spv.skl\" \
		-O3 -fopenmp \
		-DBLOCK_X=8 -DBLOCK_Y=8 \
		-I${CSDK_DIR}/usr/local/include -L${CSDK_DIR}/usr/local/lib -I./include/ \
		-Wl,-rpath -Wl,${CSDK_DIR}/usr/local/lib -std=c++17 -DCM_COMPILE_SPIRV\
		host_l0.cpp src/cpu_zncc.cpp -lze_loader -o step3.l0.skl \
		-I$(OPENCVHEADER) $(LIB_LOADPATH) $(LINK_OPENCV_LIBS) 


clean:
	rm -rf *.skl tmp*  out.jpg