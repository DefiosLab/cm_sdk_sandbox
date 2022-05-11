OPENCVHEADER = /usr/local/include/opencv4
LIB_LOADPATH = -L/usr/local/lib
LINK_OPENCV_LIBS = \
-lopencv_core \
-lopencv_imgcodecs \
-lopencv_highgui \
-lopencv_imgproc \
-lopencv_video \
-lopencv_videoio

LDFLAGS = -lm -lstdc++ $(LIB_LOADPATH) $(LINK_OPENCV_LIBS)
all : s1 s2 s3 camera

s1:
	${CSDK_DIR}/usr/bin/cmc -emit-spirv -fcmocl -mcpu=SKL -m64 \
				-isystem ${CSDK_DIR}/usr/include src/step1/kernel.cpp -o step1.spv.skl

	g++ -m64 -DKERNEL=\"step1.spv.skl\" \
		-O3 -fopenmp \
		-DBLOCK_X=1 -DBLOCK_Y=1 \
		-I${CSDK_DIR}/usr/local/include -L${CSDK_DIR}/usr/local/lib -I./include/ \
		-Wl,-rpath -Wl,${CSDK_DIR}/usr/local/lib -std=c++17 -DCM_COMPILE_SPIRV\
		src/host_l0.cpp src/cpu_zncc.cpp -lze_loader -o step1.l0.skl \
		-I$(OPENCVHEADER) $(LIB_LOADPATH) $(LINK_OPENCV_LIBS)
s2:
	${CSDK_DIR}/usr/bin/cmc -emit-spirv -fcmocl -mcpu=SKL -m64 \
				-isystem ${CSDK_DIR}/usr/include src/step2/kernel.cpp -o step2.spv.skl

	g++ -m64 -DKERNEL=\"step2.spv.skl\" \
		-O3 -fopenmp \
		-DBLOCK_X=8 -DBLOCK_Y=8 \
		-I${CSDK_DIR}/usr/local/include -L${CSDK_DIR}/usr/local/lib -I./include/ \
		-Wl,-rpath -Wl,${CSDK_DIR}/usr/local/lib -std=c++17 -DCM_COMPILE_SPIRV\
		src/host_l0.cpp src/cpu_zncc.cpp -lze_loader -o step2.l0.skl \
		-I$(OPENCVHEADER) $(LIB_LOADPATH) $(LINK_OPENCV_LIBS)

s3:
	${CSDK_DIR}/usr/bin/cmc -emit-spirv -fcmocl -mcpu=SKL -m64 \
				-isystem ${CSDK_DIR}/usr/include src/step3/kernel.cpp -o step3.spv.skl

	g++ -m64 -DKERNEL=\"step3.spv.skl\" \
		-O3 -fopenmp \
		-DBLOCK_X=8 -DBLOCK_Y=8 \
		-I${CSDK_DIR}/usr/local/include -L${CSDK_DIR}/usr/local/lib -I./include/ \
		-Wl,-rpath -Wl,${CSDK_DIR}/usr/local/lib -std=c++17 -DCM_COMPILE_SPIRV\
		src/host_l0.cpp src/cpu_zncc.cpp -lze_loader -o step3.l0.skl \
		-I$(OPENCVHEADER) $(LIB_LOADPATH) $(LINK_OPENCV_LIBS) 
camera:
	g++ src/get_temp.cpp -o get_temp \
		-I$(OPENCVHEADER) $(LIB_LOADPATH) $(LINK_OPENCV_LIBS)
	${CSDK_DIR}/usr/bin/cmc -emit-spirv -fcmocl -mcpu=SKL -m64 \
				-isystem ${CSDK_DIR}/usr/include src/step3/kernel.cpp -o step3.spv.skl

	g++ -m64 -DKERNEL=\"step3.spv.skl\" \
		-O3 -fopenmp \
		-DBLOCK_X=8 -DBLOCK_Y=8 \
		-I${CSDK_DIR}/usr/local/include -L${CSDK_DIR}/usr/local/lib -I./include/ \
		-Wl,-rpath -Wl,${CSDK_DIR}/usr/local/lib -std=c++17 -DCM_COMPILE_SPIRV\
		src/cam_run.cpp src/cpu_zncc.cpp -lze_loader -o cam_run.l0.skl \
		-I$(OPENCVHEADER) $(LIB_LOADPATH) $(LINK_OPENCV_LIBS) 
clean:
	rm -rf *.skl tmp*  out.jpg rect.png get_temp cam_run
