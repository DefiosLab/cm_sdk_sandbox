#include <iostream>
#include <cassert>
#include <math.h>
#include <vector>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <sys/time.h>
#include <float.h>
int main(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G') );
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    cv::Mat dst;
    if(!cap.isOpened())
    {
        return -1;
    }
    cv::Mat frame;
    while(cap.read(frame))
    {
        cv::Rect rect(896, 476, 128, 128);
        cv::rectangle(frame, rect, cv::Scalar(0, 0, 255), 1, cv::LINE_4);
        cv::imshow("img", frame);
        const int key = cv::waitKey(1);
        if(key == 'q')
        {
            break;
        }
        else if(key == 's')
        {
            dst = cv::Mat(frame, rect);
            cv::imwrite("rect.png", dst);
        }
    }
    cv::destroyAllWindows();
    return 0;
}
