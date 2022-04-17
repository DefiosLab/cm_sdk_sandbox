OPENCVHEADER = /usr/local/include/opencv4
LIB_LOADPATH = -L/usr/local/lib
LINK_OPENCV_LIBS = \
-lopencv_core \
-lopencv_imgcodecs \
-lopencv_highgui \
-lopencv_imgproc

LDFLAGS = -lm -lstdc++ $(LIB_LOADPATH) $(LINK_OPENCV_LIBS)

zncc: zncc.cpp ./src/cpu_zncc.cpp
	$(CXX) $^ -o $@ -I$(OPENCVHEADER) $(LIB_LOADPATH) $(LINK_OPENCV_LIBS) -I./include/  -g

clean:
	rm -rf zncc

%.o: %.cpp
	$(CXX) $(ENABLE_DEBUG) -I$(OPENCVHEADER) $(LIB_LOADPATH) $(LINK_OPENCV_LIBS)  -c $<
